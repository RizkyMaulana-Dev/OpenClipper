import json
import subprocess
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Ollama Auto-Server Wrapper")

# Konfigurasi Model
OLLAMA_MODEL = "gemma2"  # Ganti dengan model pilihanmu (misal: llama3, mistral)
OLLAMA_URL = "http://localhost:11434"


def is_ollama_running():
  """Cek apakah server Ollama sudah aktif."""
  try:
    res = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
    return res.status_code == 200
  except requests.exceptions.RequestException:
    return False


def start_ollama_service():
  """Menjalankan server Ollama di background jika belum aktif."""
  if not is_ollama_running():
    print("[+] Server Ollama belum berjalan. Memulai Ollama...")
    # Menjalankan 'ollama serve' sebagai background process
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,  # Sembunyikan jendela CMD di Windows
    )

    # Tunggu beberapa detik sampai server benar-bear siap
    for _ in range(10):
      if is_ollama_running():
        print("[+] Server Ollama berhasil aktif!")
        return True
      time.sleep(1)
    print("[-] Gagal memulai server Ollama.")
    return False
  else:
    print("[+] Server Ollama sudah aktif.")
    return True


# Jalankan pengecekan/start saat script Python pertama kali di-run
start_ollama_service()


# Format Request Data
class TranscriptRequest(BaseModel):
  chunks: list[dict]  # Format: [{"time_range": "0s - 30s", "text": "..."}]


@app.post("/analyze-highlights")
def analyze_highlights(data: TranscriptRequest):
  """Endpoint untuk mengevaluasi transkrip dan mencari momen paling lucu/viral."""
  if not is_ollama_running():
    if not start_ollama_service():
      raise HTTPException(
          status_code=500, detail="Ollama server tidak dapat diakses."
      )

  prompt = f"""
    Kamu adalah editor video klip gaming/streamer yang berpengalaman.
    Berikut adalah beberapa potongan transkrip percakapan:

    {json.dumps(data.chunks, ensure_ascii=False, indent=2)}

    Tugasmu:
    1. Analisis teks di atas.
    2. Pilih potongan yang paling LUCU, KONYOL, HYPE, atau BERPOTENSI VIRAL.
    3. Kembalikan output HANYA dalam format JSON array berisi list item terpilih.

    Format JSON Output:
    [
      {{"time_range": "10s - 40s", "reason": "Streamer kaget dan berteriak"}},
      {{"time_range": "120s - 150s", "reason": "Lelucon konyol saat collab"}}
    ]
    """

  payload = {
      "model": OLLAMA_MODEL,
      "prompt": prompt,
      "stream": False,
      "format": "json",  # Memaksa Ollama merespons dalam format JSON rapi
  }

  try:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate", json=payload, timeout=60
    )
    result = response.json()
    return json.loads(result["response"])
  except Exception as e:
    raise HTTPException(
        status_code=500, detail=f"Gagal memproses ke Ollama: {str(e)}"
    )


if __name__ == "__main__":
  import uvicorn

  # Install uvicorn dulu jika belum: pip install uvicorn
  uvicorn.run(app, host="127.0.0.1", port=8000)