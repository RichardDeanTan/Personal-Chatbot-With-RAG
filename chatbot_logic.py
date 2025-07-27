import os
import re
from dotenv import load_dotenv
import docx
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

DOC_PATH = "resource/Personal Profile - RAG purpose.docx"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
PRIMARY_LLM_MODEL = "gotocompany/gemma-2-9b-cpt-sahabatai-instruct"
VECTOR_SEARCH_TOP_K = 5
CHAT_HISTORY_WINDOW = 1

# --- 1. SETUP: Load API Key ---
def load_api_key():
    load_dotenv()
    if os.getenv("NVIDIA_API_KEY") is None:
        raise ValueError("NVIDIA_API_KEY not found. Please set it in your .env file.")
    print("API Key loaded successfully.")

# --- 2. DOCUMENT LOADING ---
def load_document(file_path):
    try:
        doc = docx.Document(file_path)
        full_text = []
        
        for para in doc.paragraphs:
            # Cek paragraf (properti item list)
            if para._p.pPr and para._p.pPr.numPr:
                indent_level = para._p.pPr.numPr.ilvl.val

                # Buat indentation
                indentation = '    ' * indent_level

                if indent_level == 0:
                    bullet = '•'
                elif indent_level == 1:
                    bullet = 'o'
                else:
                    bullet = '-'
                
                # Gabungin semuanya menjadi satu baris yang terformat
                full_text.append(f"{indentation}{bullet} {para.text}")
            else:
                full_text.append(para.text)
        
        print(f"Document '{file_path}' loaded successfully (with nested list formatting).")
        return '\n'.join(full_text)
    except FileNotFoundError:
        raise FileNotFoundError(f"The document at {file_path} was not found.")

# --- 2.1. LOGICAL CHUNKING: ---
def create_logical_chunks(text_content):
    # Regex untuk memecah teks, cth: "1. Informasi Pribadi", "2. Deskripsi Singkat"
    # Menambahkan '[A-Z]' untuk memastikan hanya memecah pada judul (yang diawali huruf kapital).
    pattern = r'\n(?=\d+\.\s[A-Z])'
    chunks = re.split(pattern, text_content)

    # Hapus chunk kosong kalo ada dan bersihin spasi di awal/akhir
    cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    if len(cleaned_chunks) > 1 and not cleaned_chunks[0].strip().startswith('1.'):
        # Gabungin judul dokumen dengan section, cth: "1. Informasi Pribadi"
        cleaned_chunks[1] = cleaned_chunks[0] + '\n\n' + cleaned_chunks[1]
        cleaned_chunks.pop(0)

    print(f"Document logically split into {len(cleaned_chunks)} semantic chunks.")
    return cleaned_chunks

# --- 3. RAG - RETRIEVAL: Create Vector Store ---
def create_vector_store(chunks):
    print(f"Creating vector store from {len(chunks)} logical chunks.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"Embedding model '{EMBEDDING_MODEL}' loaded.")

    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    print("FAISS vector store created successfully.")

    return vector_store.as_retriever(search_kwargs={'k': VECTOR_SEARCH_TOP_K})

# --- 4. RAG - GENERATION: LLM and Prompt ---
def create_llm():
    print(f"Initializing primary LLM: {PRIMARY_LLM_MODEL}")
    llm = ChatNVIDIA(
        model=PRIMARY_LLM_MODEL,
        temperature=0.7,
        top_p=0.7,
        max_tokens=256,
    )
    return llm

def create_prompt_template():
    template = """
Anda adalah 'RichBot', sebuah asisten AI dengan kepribadian yang santai, ramah, dan informatif. Anggap diri Anda seperti seorang teman yang sedang bersemangat menceritakan profil Richard Dean Tanjaya kepada pengguna.

**--- GAYA BAHASA & PERSONALITAS ---**
1.  **Santai & Tidak Kaku:** Gunakan bahasa sehari-hari yang sopan. Hindari jawaban yang terlalu formal atau terdengar seperti skrip. Boleh menggunakan kata-kata seperti "nih", "lho", "sih", "keren, kan?", atau "hehe" jika konteksnya pas.
2.  **Variasi Awalan Kalimat:** JANGAN selalu memulai jawaban dengan "Tentu!". Coba variasikan dengan "Oh, kalau soal itu...", "Oke, jadi gini...", "Siap! Untuk...", "Wah, pertanyaan bagus!", atau langsung ke poin utama.
3.  **Proaktif & Membantu:** Jika pengguna tampak bingung atau bertanya secara umum, bantu arahkan percakapan dengan menawarkan beberapa topik menarik.

**--- ATURAN UTAMA (SANGAT PENTING) ---**
1. **SELALU PERIKSA CONTEXT TERLEBIH DAHULU:** Sebelum menjawab apapun, WAJIB baca dan analisis CONTEXT yang diberikan.
2. **JANGAN PERNAH MENGARANG:** Hanya gunakan informasi yang ADA di CONTEXT. Jika tidak ada, katakan dengan jelas.
3. **KOREKSI INFORMASI YANG SALAH:** Jika pengguna menyatakan sesuatu yang BERTENTANGAN dengan CONTEXT, WAJIB koreksi dengan tegas tapi ramah.
4. **BAHASA INDONESIA 100%:** Selalu balas dalam Bahasa Indonesia, apapun bahasa pertanyaan pengguna.
5. **JANGAN SEBUT KATA "CONTEXT":** Jika informasi tidak tersedia, gunakan frasa seperti "di informasi yang aku punya", "dari yang aku tau", "berdasarkan profil Richard", atau "di data yang tersedia".

**ATURAN PALING PENTING: KOREKSI PENGGUNA JIKA SALAH**
Jika pertanyaan atau pernyataan pengguna SALAH atau BERTENTANGAN dengan `CONTEXT`, tugas utama Anda adalah MENGOREKSI mereka dengan ramah dan tegas. Jangan pernah setuju dengan informasi yang salah.

**--- SKENARIO & CONTOH WAJIB ---**

**Skenario 1: Pertanyaan Topik Baru tentang Richard**
-   *User Question:* "apa saja proyeknya?"
-   *Jawaban Ideal Anda:* "Oh, soal proyek ya? Ada beberapa yang keren nih di portofolionya, seperti Personal AI Chatbot, Prediksi Obesitas, dan Analisis Sentimen Saham. Mau aku ceritain lebih dalam soal salah satunya?"

**Skenario 2: Pertanyaan Meta ("Kamu bisa apa?")**
-   *User Question:* "kamu bisa kasi tau aku apa aja?"
-   *Jawaban Ideal Anda:* "Aku bisa ceritain banyak hal tentang Richard. Mulai dari Informasi Pribadi, Pendidikan, Keterampilan (Skills), Pengalaman kerja, Proyek-proyeknya, sampai Sertifikasi yang dia punya. Kita mulai dari mana enaknya?"

**Skenario 3: Pertanyaan Navigasi ("Bahas apa lagi?")**
-   *User Question:* "next kita mau bahas apa?" atau "apalagi yang menarik?"
-   *Jawaban Ideal Anda:* "Masih banyak lho yang bisa kita bahas! Ada topik soal Pengalaman kerja, Latar Belakang Pendidikan, atau Keterampilan teknisnya. Kamu tertarik sama yang mana?"

**Skenario 4: Pertanyaan Follow-up untuk Detail**
-   *Chat History:*
    RichBot: "...Mau aku ceritain lebih dalam soal salah satunya?"
-   *New User Question:* "jelaskan yang chatbot"
-   *Jawaban Ideal Anda:* "Oke, jadi gini. Proyek Personal AI Chatbot itu intinya dia bikin chatbot pribadi pakai teknologi AI dan RAG. Tujuannya supaya chatbotnya bisa ngasih jawaban yang pas berdasarkan data pribadinya dia. Keren, kan?" *(Setelah ini, berhenti dan tunggu respons pengguna).*

**Skenario 5: Menangani Respon Singkat/Pujian ("menarik", "hmm")**
-   *New User Question:* "menarik" atau "hmm oke"
-   *Jawaban Ideal Anda:* "Asiik, kalau kamu tertarik! Mau lanjut liat bagian lain dari profilnya, atau ada pertanyaan spesifik mungkin?"

**Skenario 6: Jawaban TIDAK ADA di Konteks**
-   *User Question:* "Dia bisa main musik gak?"
-   *Jawaban Ideal Anda:* "Wah, kalau soal kemampuan main musik, sepertinya belum ada infonya nih di profilnya. Mungkin bisa jadi hobi baru, hehe."

---
**CONTEXT:**
{context}

**CHAT HISTORY (percakapan terakhir):**
{chat_history}

**User Question:** {question}
**Jawaban Anda (dalam Bahasa Indonesia yang santai):**
"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"]
    )

# --- 5. HELPER FUNCTION: Format Chat History ---
def format_chat_history(history):
    if not history:
        return "No conversation yet."
    return "\n".join([f"User: {turn['user']}\nRichBot: {turn['bot']}" for turn in history])

# --- 6. MAIN CHAT LOGIC ---
def run_chatbot():
    load_api_key()
    document_text = load_document(DOC_PATH)
    
    logical_chunks = create_logical_chunks(document_text)

    # print("\n--- Verifikasi Hasil Logical Chunking ---")
    # for i, chunk in enumerate(logical_chunks):
    #     print(f"--- CHUNK {i+1}/{len(logical_chunks)} (Panjang: {len(chunk)} karakter) ---")
    #     print(chunk)
    #     print("====================================================\n")

    retriever = create_vector_store(logical_chunks)
    
    llm = create_llm()
    prompt_template = create_prompt_template()
    chat_history = []

    print("\n--- RichBot is Online ---")
    print("RichBot: Halo, perkenalkan namaku RichBot! Aku adalah AI Chatbot yang siap membantumu mengenal Richard. Silakan ajukan pertanyaanmu.")
    print("         (Type 'exit' to end the chat)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("RichBot: Sampai jumpa lagi!")
            break

        retrieved_docs = retriever.invoke(user_input)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

        recent_history = chat_history[-CHAT_HISTORY_WINDOW:]

        formatted_prompt = prompt_template.format(
            context=context_text,
            chat_history=format_chat_history(recent_history),
            question=user_input
        )
        
        print("RichBot: ", end="", flush=True)
        
        full_bot_response = ""
        for chunk in llm.stream(formatted_prompt):
            print(chunk.content, end="", flush=True)
            full_bot_response += chunk.content
        
        print() 

        chat_history.append({"user": user_input, "bot": full_bot_response.strip()})

if __name__ == "__main__":
    run_chatbot()