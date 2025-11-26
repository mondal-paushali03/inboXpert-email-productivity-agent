# **📬 InboXpert — AI-Powered Email Productivity Agent**

### *From Inbox Chaos to Organized Clarity*

InboXpert is an AI-driven email intelligence tool that transforms raw inbox overload into actionable insights.
It categorizes emails, extracts tasks, summarizes content, drafts professional replies, and provides a chat-based email agent — all powered by a prompt-driven architecture and an integrated SQLite backend.
<img width="1858" height="964" alt="Screenshot 2025-11-26 032620" src="https://github.com/user-attachments/assets/c4330a0d-fb3b-4b6e-84bf-bb48e65d7e75" />

**Email Details Tab**
<img width="1669" height="973" alt="image" src="https://github.com/user-attachments/assets/10cce325-a066-440e-8e3f-138cd2a4a03e" />

**Email Chat Interface**
<img width="1805" height="884" alt="image" src="https://github.com/user-attachments/assets/106fc1c1-688e-43f6-9cbf-d3fb906b8ec0" />
<img width="1696" height="939" alt="image" src="https://github.com/user-attachments/assets/e2fbff2e-0690-49f2-ae3d-c0185db3119e" />

**Draft Prompt Generator**
<img width="1687" height="932" alt="image" src="https://github.com/user-attachments/assets/f97e3b7b-302e-4313-beea-b75c6a7f40c7" />

**Email Draft Generator**
<img width="1714" height="988" alt="image" src="https://github.com/user-attachments/assets/47d2fe5f-d3ba-47a5-a470-2dcd84899995" />
<img width="1704" height="645" alt="image" src="https://github.com/user-attachments/assets/29421173-d391-48c8-a132-c644d607d327" />

---

## 🚀 **Key Features**

* **Prompt-Driven Architecture**
  Every task—categorization, summarization, action extraction, reply drafting—is handled using modular, editable prompt templates stored in SQLite.

* **Smart Email Processing Pipeline**
  Uses an LLM to process each email and attach category, summary, and extracted tasks.

* **Interactive Gradio UI**
  A clean multi-tab interface covering:

  * Inbox Viewer
  * Email Details
  * AI Email Agent Chat
  * Prompt Brain (Edit/Reset prompts)
  * Draft Composer

* **SQLite Storage Layer**
  Saves emails, action items, chat logs, drafts, and prompt templates.

* **Mock Inbox Generator**
  Comes with 18 realistic, diverse emails to demonstrate the system.

---

## 📦 **Setup Instructions**

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/mondal-paushali03/inboxpert.git
cd inboxpert
```

---

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

**Required libraries include:**

* gradio
* sqlite3
* pandas
* transformers
* torch
* pyngrok (optional for external sharing)

---

### **3️⃣ Download the Model (Qwen 2.5 – 3B Instruct)**

The backend auto-downloads the model when the script starts:

```python
model_name = "Qwen/Qwen2.5-3B-Instruct"
```

Make sure you have:

* Stable internet
* Disk space (≈5–6GB)

---

### **4️⃣ Initialize the Database**

Database setup runs automatically on launch:

```python
setup_database()
initialize_default_prompts()
```

This creates:

| Table          | Purpose                             |
| -------------- | ----------------------------------- |
| `emails`       | Stores all email content + metadata |
| `action_items` | Extracted To-Do tasks               |
| `prompts`      | Prompt templates for all LLM tasks  |
| `drafts`       | User or AI-generated email drafts   |
| `chat_history` | Agent conversation logs             |

---

## 💻 **How to Run the UI & Backend**

Simply run the complete script:

```bash
python inboxpert.py
```

The Gradio UI launches automatically:

```
http://localhost:7860
```

If ports conflict, it auto-switches to 7861 → 7862 → 7863.

```python
setup_secure_tunnel()
```

---

## 📥 **How to Load the Mock Inbox**

Mock emails are generated using:

```python
save_mock_emails()
```

What it does:

* Inserts 18 realistic emails
* Avoids duplicates
* prints how many new emails were added

You can also reset the inbox:

```python
clear_existing_emails()
save_mock_emails()
```

---

## 🧠 **How to Configure Prompts**

All prompts live in the **Prompt Brain** tab of the UI.
There you can:

✔ View
✔ Edit
✔ Update
✔ Reset to Default

Each prompt handles a specific function:

| Prompt Name           | Purpose                               |
| --------------------- | ------------------------------------- |
| `categorization`      | Important / Newsletter / Spam / To-Do |
| `action_extraction`   | JSON task extraction                  |
| `auto_reply_draft`    | AI-generated professional replies     |
| `summary_generation`  | Bullet-point summaries                |
| `email_agent_general` | Chat-based agent reasoning            |

**Updating a Prompt**

1. Open **Prompt Brain**
2. Select a template
3. Edit the text
4. Click **Save Changes**

The updated prompt is stored in SQLite + clears cache for immediate effect.

---

## 📘 **Usage Examples**

### **🔹 Process All Emails**

In UI → **Email Inbox Tab**
Click **Process All Emails**

Each email will get:

* Category
* Summary
* Extracted tasks

---

### **🔹 View Email Details**

Go to **Email Details Tab**
Select any email → Full info + AI processing available.

---

### **🔹 Use AI Email Agent**

In **Email Agent Tab**:

Ask things like:

* *“Summarize this email.”*
* *“Extract all action items.”*
* *“Draft a reply accepting the meeting.”*
* *“Why is this email important?”*

The agent responds using the prompt-driven LLM.

---

### **🔹 Generate a Smart Reply**

In **Draft Composer → Create New Draft**:

Click **AI Generated Draft**
The system:

* drafts a reply
* auto-fills subject + body
* allows you to edit
* lets you save the draft

---

### **🔹 Manage Drafts**

Under **View Saved Drafts**:

* View all saved responses
* Select → Edit
* Update or delete drafts

---

## 🏗 **System Architecture Summary**

```
User → Gradio UI
           ↓
       Backend Logic
           ↓
 SQLite Database
           ↓
       LLM Engine (Qwen)
           ↓
 Prompt Templates (editable)
```

**Processing Flow:**

1. User uploads or selects an email
2. LLM runs categorization, summarization, action extraction
3. Results saved into SQLite
4. UI displays → Agent responds → Drafts saved

---
