# 🤖 RichBot - AI Personal Assistant with RAG

Proyek ini adalah implementasi AI Personal Assistant bernama 'RichBot' yang dirancang untuk menjawab pertanyaan berdasarkan informasi personal yang disediakan dalam dokumen. RichBot memanfaatkan teknologi RAG (Retrieval-Augmented Generation) untuk memastikan jawaban yang akurat dan relevan, serta memiliki kepribadian yang santai dan ramah.

## 📂 Project Structure

- `.streamlit/config.toml` — Konfigurasi Streamlit (jika ada, untuk styling atau setting lain).
- `resource/Personal Profile - Template.docx` — Dokumen profil template untuk user download.
- `resource/PersonalProfile_RAG_purpose.docx` — Dokumen profil default yang digunakan RichBot sebagai sumber informasi.
- `.gitignore` — File untuk mengabaikan folder atau file tertentu saat push ke Git.
- `app.py` — Aplikasi Streamlit utama untuk interface web chatbot dan logika RAG.
- `chatbot_logic.py` — File yang berisi logika inti chatbot, termasuk fungsi-fungsi untuk pemrosesan dokumen, pembuatan vector store, dan interaksi dengan LLM.
- `requirements.txt` — Daftar dependensi Python yang diperlukan untuk menjalankan project.

## 🚀 Cara Run Aplikasi

### 🔹 1. Jalankan Secara Lokal
### Clone Repository
```bash
git clone https://github.com/RichardDeanTan/Personal-Chatbot-With-RAG
cd Personal-Chatbot-With-RAG
```
### Install Dependensi
```bash
pip install -r requirements.txt
```
### Konfigurasi NVIDIA API Key
Pastikan anda membuat file .streamlit/secrets.toml yang berisi NVIDIA API Key:
```bash
NVIDIA_API_KEY="YOUR_API_KEY"
```
### Jalankan Aplikasi Streamlit
```bash
streamlit run app.py
```

### 🔹 2. Jalankan Secara Online (Tidak Perlu Install)
Klik link berikut untuk langsung membuka aplikasi web:
#### 👉 [Streamlit - Personal Chatbot with RAG](https://personal-chatbot-with-rag-richardtanjaya.streamlit.app/)

## 💡 Fitur
**✅ Personalized Q&A |** Menjawab pertanyaan pengguna berdasarkan profil personal yang disediakan.
**✅ Retrieval-Augmented Generation (RAG) |** Memastikan akurasi jawaban dengan mengambil informasi dari dokumen relevan.
**✅ Custom Document Upload |** Pengguna dapat mengunggah dokumen profil mereka sendiri (format .docx) untuk digunakan RichBot.
**✅ Dynamic Personalities |** RichBot dirancang dengan kepribadian santai, ramah, dan informatif.
**✅ Session-based Memory |** Mengelola riwayat percakapan untuk konteks (memori terbatas pada 1 interaksi terakhir).
**✅ Customizable Parameters |** Mengatur parameter LLM seperti Temperature, Top P, Max Tokens, dan jumlah dokumen yang diambil (K).
**✅ Interactive Web Interface |** User-friendly interface menggunakan Streamlit dengan styling kustom.

## ⚙️ Tech Stack
- **Large Language Model (LLM)** ~ NVIDIA NIM (via `langchain_nvidia_ai_endpoints`)
- **Retrieval** ~ FAISS (sebagai Vector Store)
- **Embeddings** ~ HuggingFace Embeddings (`sentence-transformers`)
- **Framework** ~ LangChain
- **Web Framework** ~ Streamlit
- **Document Processing** ~ python-docx
- **Deployment** ~ Streamlit Cloud

## 🧠 Model Details
- **Primary LLM Model** ~ `gotocompany/gemma-2-9b-cpt-sahabatai-instruct`
- **Embedding Model** ~ `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **Task** ~ Question Answering dengan RAG
- **Language** ~ Indonesian (Bahasa Indonesia)
- **Key Techniques** ~ RAG, Prompt Engineering, Session Management

## ⭐ Deployment
Aplikasi ini di-deploy menggunakan:
- Streamlit Cloud
- GitHub

## 👨‍💻 Pembuat
Richard Dean Tanjaya

## 📝 License
Proyek ini bersifat open-source dan bebas digunakan untuk keperluan edukasi dan penelitian.