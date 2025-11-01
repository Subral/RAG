# 🧠 RAG (Retrieval-Augmented Generation) Application

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **Ollama** for local LLM and embedding inference. The app retrieves relevant context from a document store and generates intelligent responses using a fine-tuned language model.

---

## 🚀 Features

* 🔍 **Context-Aware Retrieval** using the `bge-base-en-v1.5` embedding model
* 💬 **Natural Language Generation** powered by `Llama-3.2-1B-Instruct`
* 🧩 **Local Execution** — fully offline, no external API required
* ⚡ **Fast and Lightweight** — ideal for local experimentation and private deployments

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/subral/RAG.git
cd RAG
```

### 2. Install Ollama

Download and install **Ollama** from the official site:
👉 [https://ollama.com/download](https://ollama.com/download)

---

## 📦 Model Setup

After installing Ollama, pull the required models:

### 🧩 Embedding Model

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
```

Model Source: [CompendiumLabs/bge-base-en-v1.5-gguf](https://huggingface.co/CompendiumLabs/bge-base-en-v1.5-gguf)

### 🧠 Language Model

```bash
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

Model Source: [bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF)

---

## 🧩 Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the app with:

```bash
python app.py
```

Then open your browser at:

```
http://localhost:5000
```

---

## 🧠 How It Works

1. **User Query:** The user submits a question or prompt.
2. **Retriever:** The system embeds the query using the embedding model and retrieves the most relevant chunks.
3. **Generator:** The language model uses the retrieved context to generate a factual and coherent response.

---

## 📁 Project Structure

```
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── data/                  # Folder for documents or datasets
├── models/                # (Optional) Custom models or configs
└── README.md              # Project documentation
```

---

## ⚙️ Requirements

* Python 3.12
* Ollama installed and configured
* Internet connection (for first-time model download)

---

## 🧑‍💻 Author

**Subral Jaiswal**
💼 GitHub: [@yourusername](https://github.com/subral)
📧 Email: [your.email@example.com](subraljaiswal6@gmail.com)

---

## 🪶 License

This project is licensed under the [MIT License](LICENSE).
