# Cell 1: All Imports and Database Setup

import gradio as gr
import sqlite3
import json
import uuid
from datetime import datetime
import pandas as pd
import re
import os
from typing import List, Dict, Any

# Global variables
DATABASE_NAME = 'email_agent.db'

def setup_database():
    """Initialize SQLite database with all required tables"""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # Emails table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emails (
            id TEXT PRIMARY KEY,
            sender TEXT,
            subject TEXT,
            body TEXT,
            timestamp TEXT,
            category TEXT,
            priority TEXT,
            summary TEXT,
            draft_reply TEXT,
            is_processed BOOLEAN DEFAULT FALSE,
            created_at TEXT
        )
    ''')

    # Action items table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS action_items (
            id TEXT PRIMARY KEY,
            email_id TEXT,
            task TEXT,
            deadline TEXT,
            priority TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT,
            FOREIGN KEY (email_id) REFERENCES emails (id)
        )
    ''')

    # Prompts table - Core of the prompt-driven architecture
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prompts (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE,
            content TEXT,
            description TEXT,
            category TEXT,
            version INTEGER DEFAULT 1,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TEXT,
            updated_at TEXT
        )
    ''')

    # Chat history for email agent
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id TEXT PRIMARY KEY,
            email_id TEXT,
            user_query TEXT,
            agent_response TEXT,
            context_used TEXT,
            created_at TEXT,
            FOREIGN KEY (email_id) REFERENCES emails (id)
        )
    ''')

    # Drafts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS drafts (
            id TEXT PRIMARY KEY,
            email_id TEXT,
            subject TEXT,
            body TEXT,
            recipient TEXT,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY (email_id) REFERENCES emails (id)
        )
    ''')

    conn.commit()
    conn.close()
    print("Database setup completed!")

# Initialize database
setup_database()



# Cell 2: Default Prompt Templates Storage

def initialize_default_prompts():
    """Initialize all default prompt templates as per problem requirements"""
    default_prompts = [
        {
            "name": "categorization",
            "content": """Categorize the following email into exactly one of these categories: Important, Newsletter, Spam, To-Do.

            Email Details:
            From: {sender}
            Subject: {subject}
            Body: {body}

            Rules:
            - To-Do emails must include a direct request requiring user action
            - Newsletter emails typically contain mass distribution content, unsubscribe links
            - Spam emails contain suspicious links, prize notifications, or urgent financial requests
            - Important emails are work-related, personal, or require attention but no immediate action

            Respond ONLY with the category name: Important, Newsletter, Spam, or To-Do.""",
            "description": "Email categorization into predefined categories",
            "category": "classification"
        },
        {
            "name": "action_extraction",
            "content": """Extract all actionable tasks from the following email. Respond in valid JSON format only:

            Email: {body}

            Required JSON structure:
            {{
                "action_items": [
                    {{
                        "task": "clear description of what needs to be done",
                        "deadline": "extracted deadline or 'Not specified'",
                        "priority": "High/Medium/Low based on urgency language"
                    }}
                ]
            }}

            Extract deadlines from phrases like "by Friday", "deadline is", "due tomorrow".
            Determine priority: High for urgent/immediate, Medium for soon, Low for whenever.""",
            "description": "Action item extraction in JSON format",
            "category": "extraction"
        },
        {
            "name": "auto_reply_draft",
            "content": """Draft a professional email reply based on the original email and context.

            Original Email:
            From: {sender}
            Subject: {subject}
            Body: {body}

            User Context: {user_context}

            Guidelines:
            - Maintain professional tone
            - Address the sender appropriately
            - Respond to key points in the original email
            - Keep it concise (3-5 sentences)
            - Do NOT include placeholders like [Name]
            - If it's a meeting request, ask for an agenda
            - If it's a task request, acknowledge and provide timeline
            - Sign off appropriately

            Draft the reply:""",
            "description": "Auto-reply drafting for emails",
            "category": "generation"
        },
        {
            "name": "summary_generation",
            "content": """Summarize the following email in 2-3 clear bullet points focusing on key information and requests.

            Email:
            From: {sender}
            Subject: {subject}
            Body: {body}

            Create a concise summary that captures:
            - Main purpose of the email
            - Key information shared
            - Any requests or actions needed
            - Important deadlines or dates

            Format as bullet points:""",
            "description": "Email summarization in bullet points",
            "category": "summarization"
        },
        {
            "name": "email_agent_general",
            "content": """You are an AI email assistant. Help the user with their email query.

            Email Context:
            From: {sender}
            Subject: {subject}
            Body: {body}

            User Query: {user_query}

            Available Actions:
            - Summarize the email
            - Extract action items
            - Categorize the email
            - Draft replies
            - Answer questions about the content

            Provide helpful, accurate information based on the email content.
            If the query is unclear, ask for clarification.

            Response:""",
            "description": "General email agent for chat interactions",
            "category": "assistant"
        }
    ]

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    for prompt in default_prompts:
        cursor.execute('''
            INSERT OR REPLACE INTO prompts (id, name, content, description, category, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()),
            prompt['name'],
            prompt['content'],
            prompt['description'],
            prompt['category'],
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))

    conn.commit()
    conn.close()
    print("Default prompts initialized!")

# Initialize prompts
initialize_default_prompts()



# Cell 3: Mock Inbox Data

def create_mock_emails():
    """Create comprehensive mock inbox with 18 diverse emails as per problem requirements"""
    return [
        # URGENT & ACTION REQUIRED
        {
            "sender": "ceo@company.com",
            "subject": "URGENT: Quarterly Board Meeting Preparation",
            "body": """Team,

We have the quarterly board meeting this Friday and I need the following prepared:

REQUIRED BY THURSDAY EOD:
1. Q3 Financial Performance Dashboard - Sarah
2. Client Satisfaction Metrics - Mike
3. Product Roadmap Update - David
4. Risk Assessment Report - Lisa

This is absolutely critical. The board will make funding decisions based on these materials.

Please coordinate and send me drafts by Thursday 5 PM.

Best,
CEO""",
            "timestamp": "2024-01-15 09:00:00"
        },
        {
            "sender": "hr@company.com",
            "subject": "ACTION REQUIRED: Benefits Enrollment Deadline",
            "body": """Dear Employee,

This is a final reminder that the annual benefits enrollment period closes TOMORROW at 5:00 PM.

You must:
1. Log into the HR portal and review your options
2. Select your health, dental, and retirement plans
3. Submit your electronic confirmation
4. Save your confirmation for records

Failure to enroll will result in default basic coverage.

Visit: hrportal.company.com/benefits

HR Department""",
            "timestamp": "2024-01-14 14:30:00"
        },

        # NEWSLETTERS
        {
            "sender": "news@techdigest.com",
            "subject": "Tech Digest Weekly: AI Innovations & Market Trends",
            "body": """TECH DIGEST WEEKLY

Featured This Week:
AI & Machine Learning
- New transformer models breakthrough
- Open-source AI tools update
- Ethics in AI development

Industry Insights
- Cloud computing trends 2024
- Remote work infrastructure
- Cybersecurity updates

Developer Corner
- New framework releases
- Best practices for 2024
- Community events

Unsubscribe: [link] | Preferences: [link]

Tech Digest Team""",
            "timestamp": "2024-01-15 08:00:00"
        },
        {
            "sender": "marketing@industrynews.com",
            "subject": "Weekly Marketing Insights & Strategies",
            "body": """MARKETING INSIGHTS NEWSLETTER

This week's highlights:
Performance Metrics
- Email engagement trends
- Social media algorithm changes
- Conversion rate optimizations

Campaign Strategies
- Personalization techniques
- Multi-channel approaches
- ROI measurement

Industry Benchmarks
- Competitor analysis
- Market positioning
- Growth opportunities

To unsubscribe from these emails: click here
To update preferences: click here

Marketing Insights Team""",
            "timestamp": "2024-01-14 11:00:00"
        },

        # MEETING REQUESTS
        {
            "sender": "project.manager@company.com",
            "subject": "Meeting: Project Alpha Kickoff",
            "body": """Hi Team,

Let's schedule a kickoff meeting for Project Alpha next week. I'd like to discuss:

- Project scope and objectives
- Timeline and milestones
- Resource allocation
- Risk mitigation strategies

Please let me know your availability for:
- Monday 10 AM
- Tuesday 2 PM
- Wednesday 11 AM

The meeting should take about 1 hour. We'll use Conference Room B or Teams.

Best,
Project Manager""",
            "timestamp": "2024-01-13 16:45:00"
        },
        {
            "sender": "client@importantclient.com",
            "subject": "Urgent: System Issue Discussion",
            "body": """Support Team,

We're experiencing critical system errors after the latest deployment. Can we schedule an emergency call today to discuss:

1. Immediate mitigation steps
2. Root cause analysis
3. Communication plan to our users
4. Long-term prevention

Our team is available:
- 10:00 AM - 12:00 PM
- 2:00 PM - 4:00 PM

This is blocking our production environment.

Thanks,
Client Director""",
            "timestamp": "2024-01-15 08:30:00"
        },

        # TASK REQUESTS
        {
            "sender": "team.lead@company.com",
            "subject": "Code Review Required for Authentication Module",
            "body": """Development Team,

The new authentication module is ready for review. Please review the following PRs by EOD tomorrow:

PR #245 - OAuth implementation
PR #246 - Password security updates
PR #247 - Session management

Focus on:
- Security vulnerabilities
- Code quality and standards
- Performance implications
- API consistency

Reviewers: Alice, Bob, Carol

Please provide detailed feedback in the PR comments.

Thanks,
Team Lead""",
            "timestamp": "2024-01-14 13:15:00"
        },
        {
            "sender": "colleague@company.com",
            "subject": "Help needed with database migration",
            "body": """Hi,

I'm working on the database migration for the customer portal and could use your expertise.

Could you:
1. Review the migration script I've attached?
2. Help test the data integrity after migration?
3. Suggest any optimizations for the process?

I'm aiming to complete this by Friday. Let me know when you might have 30 minutes to walk through it.

Thanks!""",
            "timestamp": "2024-01-14 10:20:00"
        },

        # SPAM
        {
            "sender": "prize@international-lottery.com",
            "subject": "CONGRATULATIONS! You Won $2,500,000!",
            "body": """OFFICIAL WINNING NOTIFICATION

Dear Winner,

You have been selected as the Grand Prize winner of $2,500,000 in our international lottery!

To claim your prize, you must:
1. Click here: http://suspicious-link-1.com/claim
2. Provide your banking information for transfer
3. Pay the $500 processing fee
4. Verify your identity documents

This exclusive offer expires in 24 hours! Don't miss this life-changing opportunity!

Sincerely,
International Lottery Commission""",
            "timestamp": "2024-01-15 07:30:00"
        },
        {
            "sender": "security@bank-alert.com",
            "subject": "URGENT: Suspicious Activity Detected on Your Account",
            "body": """SECURITY ALERT

We detected unusual login activity from a new device in a different location.

If this wasn't you, please:
- Verify your account immediately: http://fake-bank-security.com/verify
- Update your security questions
- Review recent transactions

Failure to act within 12 hours will result in account suspension.

This is an automated message - please do not reply.

Bank Security Team""",
            "timestamp": "2024-01-14 15:45:00"
        },

        # PROJECT UPDATES
        {
            "sender": "updates@project-tracker.com",
            "subject": "Project Phoenix Weekly Status Update",
            "body": """PROJECT PHOENIX - WEEKLY UPDATE

Current Status: On Track
Completion: 65%

This Week's Accomplishments:
User authentication module completed
Database optimization implemented
Frontend responsive design finalized

Next Week's Priorities:
API integration testing
Performance benchmarking
Documentation updates

Risks:
- Third-party API delays (Medium risk)
- Resource constraints (Low risk)

The project remains on schedule for March 1st delivery.

Project Manager""",
            "timestamp": "2024-01-12 09:00:00"
        },
        {
            "sender": "engineering@company.com",
            "subject": "System Maintenance Completed Successfully",
            "body": """SYSTEM MAINTENENACE COMPLETION NOTICE

The scheduled system maintenance has been completed successfully.

Completed Work:
- Database server upgrades
- Security patch deployments
- Performance optimizations
- Backup system verification

All systems are now operational. No issues were encountered during the maintenance window.

Next maintenance scheduled for: February 15, 2024, 2:00 AM - 4:00 AM

Engineering Team""",
            "timestamp": "2024-01-13 06:00:00"
        },

        # PERSONAL
        {
            "sender": "friend@personal.com",
            "subject": "Lunch next week?",
            "body": """Hey!

Long time no see! Are you free for lunch next week? I'd love to catch up and hear how things are going.

I'm available:
- Tuesday around 12:30
- Wednesday after 1:00
- Thursday anytime

How about that new Italian place downtown? I've heard great things.

Let me know what works for you!

Cheers,
Alex""",
            "timestamp": "2024-01-14 12:30:00"
        },
        {
            "sender": "family@personal.com",
            "subject": "Weekend plans",
            "body": """Hi,

Just checking in about the weekend. Are we still on for dinner on Saturday?

Mom said she's making your favorite dessert. We should plan to arrive around 6 PM.

Also, could you bring that board game everyone liked last time?

Let me know if the timing works.

Love,
Family""",
            "timestamp": "2024-01-13 18:00:00"
        },

        # CLIENT COMMUNICATIONS
        {
            "sender": "important.client@corporate.com",
            "subject": "Contract Renewal Discussion",
            "body": """Hello,

Our current contract is up for renewal next month. I'd like to schedule a call to discuss:

1. Contract terms for the next year
2. Service level agreements
3. Pricing structure
4. Additional services needed

Our team is available:
- Monday, January 22: 10 AM - 12 PM
- Tuesday, January 23: 2 PM - 4 PM
- Wednesday, January 24: 9 AM - 11 AM

Please let me know which time slot works best for your team.

Looking forward to continuing our partnership.

Best regards,
Client Relations Manager""",
            "timestamp": "2024-01-12 14:15:00"
        },
        {
            "sender": "vendor@supplier.com",
            "subject": "Invoice #INV-78234 Past Due - Immediate Attention Required",
            "body": """PAST DUE NOTICE - URGENT

Invoice #INV-78234 for services rendered is now 60 days past due.

Invoice Details:
- Amount: $23,450.00
- Services: Cloud Infrastructure Q4 2024
- Due Date: November 15, 2024
- Current Status: SIGNIFICANTLY OVERDUE

Immediate payment is required to avoid:
- Service interruption effective immediately
- Late fees accruing at 5% monthly
- Credit hold on your account

Please process payment today or contact accounts@supplier.com to discuss payment arrangements.

Accounts Receivable Department""",
            "timestamp": "2024-01-15 10:00:00"
        },

        # INTERNAL UPDATES
        {
            "sender": "facilities@company.com",
            "subject": "Office Renovation Starting Monday",
            "body": """OFFICE UPDATE NOTICE

The 3rd floor renovation begins next Monday and will continue for 3 weeks.

Affected Areas:
- North wing bathrooms (completely closed)
- Main kitchen area (limited access)
- Conference rooms B & C (unavailable)

Temporary Arrangements:
- Portable facilities on 2nd floor
- Mobile kitchenette in break area
- Book alternative meeting spaces in advance

We apologize for any inconvenience. The result will be modernized, more comfortable workspaces.

Facilities Management""",
            "timestamp": "2024-01-12 11:30:00"
        },
        {
            "sender": "social.committee@company.com",
            "subject": "Holiday Party Photos Available",
            "body": """HOLIDAY PARTY PHOTOS

The photos from our annual holiday party are now available!

You can:
- View all photos at: companyportal.com/events/holiday2024
- Download your favorites
- Share with colleagues
- Order prints if desired

A big thank you to everyone who attended and made it a wonderful event!

Social Committee""",
            "timestamp": "2024-01-11 13:45:00"
        }
    ]

def save_mock_emails():
    """Save mock emails to database with duplicate prevention"""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # First, let's check what's currently in the database
    cursor.execute('SELECT COUNT(*) FROM emails')
    current_count = cursor.fetchone()[0]
    print(f"Current emails in database: {current_count}")

    # If we already have emails, check if they match our mock data
    if current_count > 0:
        cursor.execute('SELECT sender, subject, timestamp FROM emails LIMIT 5')
        existing_samples = cursor.fetchall()
        print(f"Sample existing emails: {existing_samples[:3]}")

    mock_emails = create_mock_emails()
    saved_count = 0
    skipped_count = 0

    for email in mock_emails:
        try:
            # Check if this exact email already exists (by sender, subject, and timestamp)
            cursor.execute(
                'SELECT id FROM emails WHERE sender = ? AND subject = ? AND timestamp = ?',
                (email['sender'], email['subject'], email['timestamp'])
            )
            existing_email = cursor.fetchone()

            if existing_email:
                skipped_count += 1
                continue

            # Insert only if it doesn't exist
            email_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO emails (id, sender, subject, body, timestamp, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                email_id,
                email['sender'],
                email['subject'],
                email['body'],
                email['timestamp'],
                datetime.now().isoformat()
            ))
            saved_count += 1

        except Exception as e:
            print(f"Failed to save email from {email['sender']}: {e}")

    conn.commit()

    # Get final count
    cursor.execute('SELECT COUNT(*) FROM emails')
    final_count = cursor.fetchone()[0]
    conn.close()

    print(f"Saved {saved_count} new mock emails to database!")
    print(f"Skipped {skipped_count} duplicate emails")
    print(f"Total emails in database: {final_count}")

    return saved_count

def clear_existing_emails():
    """Clear all existing emails from database (use carefully)"""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM emails')
    before_count = cursor.fetchone()[0]

    if before_count > 0:
        cursor.execute('DELETE FROM emails')
        cursor.execute('DELETE FROM action_items')
        cursor.execute('DELETE FROM chat_history')
        cursor.execute('DELETE FROM drafts')

        conn.commit()
        print(f"Cleared {before_count} existing emails and related data")
    else:
        print("No existing emails to clear")

    conn.close()


# Create and save mock emails
print("Loading mock emails...")
saved_count = save_mock_emails()
print(f"Mock inbox ready with {saved_count} new emails")



# Cell-4: LLM Integration

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from typing import Dict, Any, List
import json

class QwenLLMEngine:
    """LLM engine using Qwen for all tasks"""

    def __init__(self):
        self.prompt_cache = {}
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Using device: {self.device}")
        self._load_model()

    def _load_model(self):
        """Load Qwen model for all tasks"""
        try:
            # Load Qwen for all tasks
            print("Loading Qwen model...")
            model_name = "Qwen/Qwen2.5-3B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left"
            )

            # Set pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("Qwen model loaded successfully!")

        except Exception as e:
            print(f"Failed to load Qwen model: {e}")
            # Fallback to pipeline if direct loading fails
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model=model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    pad_token_id=151643
                )
                self.use_pipeline = True
                print("Qwen pipeline loaded as fallback!")
            except Exception as e2:
                print(f"Failed to load Qwen fallback: {e2}")
                self.use_pipeline = False

    def get_prompt(self, prompt_name: str) -> str:
        """Get prompt from database with caching"""
        if prompt_name in self.prompt_cache:
            return self.prompt_cache[prompt_name]

        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute('SELECT content FROM prompts WHERE name = ?', (prompt_name,))
        result = cursor.fetchone()
        conn.close()

        if result:
            self.prompt_cache[prompt_name] = result[0]
            return result[0]
        return None

    def fill_prompt_template(self, prompt: str, **kwargs) -> str:
        """Fill prompt template with actual values"""
        filled_prompt = prompt
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            filled_prompt = filled_prompt.replace(placeholder, str(value))
        return filled_prompt

    def call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """Call Qwen model for all tasks"""
        try:
            if hasattr(self, 'use_pipeline') and self.use_pipeline:
                return self._call_pipeline(prompt, max_tokens)
            elif self.model:
                return self._call_model_direct(prompt, max_tokens)
            else:
                return "Model not available"
        except Exception as e:
            print(f"LLM call failed: {e}")
            return "Please try the operation again."

    def _call_model_direct(self, prompt: str, max_tokens: int) -> str:
        """Call Qwen model directly with proper attention mask"""
        # Format prompt for Qwen
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize with attention mask
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        return response

    def _call_pipeline(self, prompt: str, max_tokens: int) -> str:
        """Call Qwen using pipeline"""
        try:
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

            result = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else 151643,
                eos_token_id=self.tokenizer.eos_token_id if self.tokenizer else 151645
            )
            response = result[0]['generated_text'].strip()
            # Extract only the assistant's response
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()
            return response
        except Exception as e:
            print(f"Pipeline call failed: {e}")
            return "Please try the operation again."

# Initialize the LLM engine
print("Initializing Qwen LLM Engine...")
llm_engine = QwenLLMEngine()
print("Qwen LLM Engine ready!")



# Cell-5: E-Mail Processing Pipeline

def process_single_email(email_id: str) -> Dict[str, Any]:
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # Get email data
    cursor.execute('SELECT sender, subject, body FROM emails WHERE id = ?', (email_id,))
    email_data = cursor.fetchone()

    if not email_data:
        return {"error": "Email not found"}

    sender, subject, body = email_data

    # PROPER PROMPT-DRIVEN CATEGORIZATION WITH QWEN
    categorization_prompt_content = llm_engine.get_prompt("categorization")
    if categorization_prompt_content:
        filled_prompt = llm_engine.fill_prompt_template(
            categorization_prompt_content,
            sender=sender,
            subject=subject,
            body=body
        )

        category = llm_engine.call_llm(filled_prompt, max_tokens=50)
        # Clean up category response
        category = category.split('\n')[0].strip()

    else:
        category = "Important"

    # PROPER PROMPT-DRIVEN ACTION EXTRACTION WITH QWEN
    action_prompt_content = llm_engine.get_prompt("action_extraction")
    action_items_json = '{"action_items": []}'
    if action_prompt_content:
        filled_prompt = llm_engine.fill_prompt_template(
            action_prompt_content,
            body=body
        )

        action_items_json = llm_engine.call_llm(filled_prompt, max_tokens=200)

    # PROPER PROMPT-DRIVEN SUMMARIZATION WITH QWEN
    summary_prompt_content = llm_engine.get_prompt("summary_generation")
    summary = ""
    if summary_prompt_content:
        filled_prompt = llm_engine.fill_prompt_template(
            summary_prompt_content,
            sender=sender,
            subject=subject,
            body=body
        )

        summary = llm_engine.call_llm(filled_prompt, max_tokens=150)

    # Update email with processed data - Always update even if processed before
    cursor.execute('''
        UPDATE emails
        SET category = ?, summary = ?, is_processed = TRUE
        WHERE id = ?
    ''', (category.strip(), summary.strip(), email_id))

    # Clear existing action items for this email before adding new ones
    cursor.execute('DELETE FROM action_items WHERE email_id = ?', (email_id,))

    # Save action items from LLM response
    action_items = []
    try:
        # Try to parse JSON response
        action_data = json.loads(action_items_json)
        action_items = action_data.get("action_items", [])
    except json.JSONDecodeError:
        # Extract actions from text response
        lines = action_items_json.split('\n')
        for line in lines:
            if line.strip() and any(keyword in line.lower() for keyword in ['task', 'action', 'review', 'prepare', 'meeting', 'help']):
                action_items.append({
                    "task": line.strip(),
                    "deadline": "Not specified",
                    "priority": "Medium"
                })

    # Save action items to database
    for action in action_items:
        action_id = str(uuid.uuid4())
        # Ensure we're working with a dictionary
        if not isinstance(action, dict):
            action = {"task": str(action), "deadline": "Not specified", "priority": "Medium"}

        cursor.execute('''
            INSERT INTO action_items (id, email_id, task, deadline, priority, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            action_id, email_id,
            action.get('task', 'Review email'),
            action.get('deadline', 'Not specified'),
            action.get('priority', 'Medium'),
            datetime.now().isoformat()
        ))

    conn.commit()
    conn.close()

    return {
        "email_id": email_id,
        "category": category,
        "summary": summary,
        "action_items": action_items
    }

def process_all_emails():
    """Process all emails using Qwen LLM (processes all emails every time)"""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # Get ALL emails, not just unprocessed ones
    cursor.execute('SELECT id FROM emails')
    all_emails = cursor.fetchall()
    conn.close()

    print(f"Processing {len(all_emails)} emails...")

    results = []
    for (email_id,) in all_emails:
        try:
            result = process_single_email(email_id)
            results.append(result)
        except Exception as e:
            print(f"Failed to process email {email_id}: {e}")

    return results

# Process all mock emails with Qwen LLM
print("Starting email processing with Qwen LLM...")
processing_results = process_all_emails()
print(f"Processing completed. Results: {len(processing_results)} emails processed")



# Cell 6: Email Agent Chat System

def chat_with_email_agent(email_id: str, user_query: str) -> str:
    """Chat interface for email agent using open-source LLM"""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # Get email data
    cursor.execute('SELECT sender, subject, body, category, summary FROM emails WHERE id = ?', (email_id,))
    email_data = cursor.fetchone()
    conn.close()

    if not email_data:
        return "Email not found. Please select a valid email."

    sender, subject, body, category, summary = email_data

    # Create a more specific prompt based on the query type
    if "summar" in user_query.lower():
        prompt_template = """Create a comprehensive summary of this email:

Email Details:
From: {sender}
Subject: {subject}
Body: {body}

Please provide:
1. Main purpose and key points
2. Important information shared
3. Action items or requests
4. Deadlines or important dates

Summary:"""

    elif "action" in user_query.lower() or "task" in user_query.lower() or "extract" in user_query.lower():
        prompt_template = """Extract all action items and tasks from this email:

Email Details:
From: {sender}
Subject: {subject}
Body: {body}

Please list all actionable items with:
- Specific tasks that need to be done
- Deadlines or timeframes mentioned
- Priority level (High/Medium/Low)
- Responsible parties if mentioned

Action Items:"""

    elif "categor" in user_query.lower():
        prompt_template = """Categorize this email and explain your reasoning:

Email Details:
From: {sender}
Subject: {subject}
Body: {body}

Categories: Important, Newsletter, Spam, To-Do

Analysis and Category:"""

    elif "reply" in user_query.lower() or "draft" in user_query.lower() or "respond" in user_query.lower():
        prompt_template = """Draft a professional reply to this email:

Original Email:
From: {sender}
Subject: {subject}
Body: {body}

Please write a professional response that:
- Addresses the main points
- Is concise and clear
- Maintains appropriate tone
- Provides necessary information

Draft Reply:"""

    else:
        # General email agent prompt
        prompt_template = """You are an AI email assistant. Based on the email below, answer the user's question.

Email Context:
From: {sender}
Subject: {subject}
Body: {body}
Current Category: {category}
Summary: {summary}

User Question: {user_query}

Please provide a helpful response based on the email content:"""

    # Fill the prompt template
    if "categor" in user_query.lower():
        filled_prompt = llm_engine.fill_prompt_template(
            prompt_template,
            sender=sender,
            subject=subject,
            body=body[:800]
        )
    elif "summar" in user_query.lower() or "action" in user_query.lower() or "reply" in user_query.lower():
        filled_prompt = llm_engine.fill_prompt_template(
            prompt_template,
            sender=sender,
            subject=subject,
            body=body[:800]
        )
    else:
        filled_prompt = llm_engine.fill_prompt_template(
            prompt_template,
            sender=sender,
            subject=subject,
            body=body[:800],
            category=category,
            summary=summary,
            user_query=user_query
        )

    print("Sending chat query to LLM...")
    print(f"Query type: {user_query}")
    response = llm_engine.call_llm(filled_prompt, max_tokens=500)


    # Save chat history
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_history (id, email_id, user_query, agent_response, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        str(uuid.uuid4()),
        email_id,
        user_query,
        response,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

    return response

def generate_smart_reply(email_id: str, user_context: str = "Professional workplace context") -> str:
    """Generate smart reply using open-source LLM"""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute('SELECT sender, subject, body FROM emails WHERE id = ?', (email_id,))
    email_data = cursor.fetchone()
    conn.close()

    if not email_data:
        return "Email not found."

    sender, subject, body = email_data

    # Get the auto-reply prompt
    reply_prompt = llm_engine.get_prompt("auto_reply_draft")
    if not reply_prompt:
        reply_prompt = """Draft a professional reply to this email:

        Original Email:
        From: {sender}
        Subject: {subject}
        Body: {body}

        Context: {user_context}

        Write a concise, professional response that addresses the main points."""

    filled_prompt = llm_engine.fill_prompt_template(
        reply_prompt,
        sender=sender,
        subject=subject,
        body=body[:800],
        user_context=user_context
    )

    print("Generating smart reply with LLM...")
    reply = llm_engine.call_llm(filled_prompt, max_tokens=400)

    # Save draft to database
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE emails SET draft_reply = ? WHERE id = ?
    ''', (reply, email_id))
    conn.commit()
    conn.close()

    return reply

def get_email_action_items(email_id: str) -> List[Dict]:
    """Get action items for an email"""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT task, deadline, priority, status
        FROM action_items
        WHERE email_id = ?
    ''', (email_id,))

    actions = []
    for task, deadline, priority, status in cursor.fetchall():
        actions.append({
            "task": task,
            "deadline": deadline,
            "priority": priority,
            "status": status
        })

    conn.close()
    return actions



# Cell-7: Deployment Setup

from pyngrok import ngrok
from google.colab import userdata
import getpass

def setup_secure_tunnel():
    """Set up secure ngrok tunnel with token handling"""
    try:
        NGROK_TOKEN = userdata.get('NGROK_TOKEN')
        print("✅ Token loaded from Colab secrets")
    except:
        # Fallback: manual input
        print("🔐 Enter your ngrok token (get free one from https://ngrok.com):")
        NGROK_TOKEN = getpass.getpass('Token: ')

    # Set up ngrok
    ngrok.set_auth_token(NGROK_TOKEN)
    public_url = ngrok.connect(7860)

    print(f"🚀 Secure tunnel established!")
    print(f"🌐 Your app will be available at: {public_url}")
    return public_url



# Cell-8: UI Interface

def create_complete_gradio_interface():

    with gr.Blocks(theme=gr.themes.Soft(), title="InboXpert") as demo:
        gr.Markdown("# InboXpert")
        gr.Markdown("### From Inbox Chaos to Organized Clarity")

        # State variables
        email_choices_state = gr.State([])
        email_ids_state = gr.State([])
        current_draft_id_state = gr.State("")

        # ===== INBOX TAB =====
        with gr.Tab("Email Inbox"):
            gr.Markdown("### Email Inbox Viewer & Processor")

            with gr.Row():
                with gr.Column(scale=1):
                    category_filter = gr.Dropdown(
                        choices=["All", "Important", "Newsletter", "Spam", "To-Do"],
                        value="All",
                        label="Filter by Category",
                        interactive=True
                    )
                    refresh_inbox_btn = gr.Button("Refresh Inbox", variant="primary")
                    process_all_btn = gr.Button("Process All Emails", variant="secondary")

                    # Stats
                    stats_display = gr.Textbox(
                        label="Inbox Statistics",
                        lines=3,
                        interactive=False
                    )

                with gr.Column(scale=3):
                    inbox_display = gr.Dataframe(
                        label="Emails",
                        headers=["Sender", "Subject", "Timestamp", "Category", "Summary"],
                        datatype=["str", "str", "str", "str", "str"],
                        row_count=10,
                        col_count=(5, "fixed"),
                        interactive=False
                    )

        # ===== EMAIL DETAILS TAB =====
        with gr.Tab("Email Details"):
            gr.Markdown("### Email Details & Processing")

            with gr.Row():
                email_selector = gr.Dropdown(
                    label="Select Email",
                    choices=[],
                    interactive=True
                )
                load_emails_btn = gr.Button("Load Emails")

            with gr.Row():
                email_info = gr.Textbox(
                    label="Email Information",
                    lines=8,
                    interactive=False
                )

            with gr.Row():
                email_body = gr.Textbox(
                    label="Email Body",
                    lines=10,
                    interactive=False
                )

            with gr.Row():
                process_btn = gr.Button("Process with AI", variant="primary")
                category_output = gr.Textbox(label="Category", interactive=False)
                summary_output = gr.Textbox(label="Summary", lines=4, interactive=False)

        # ===== EMAIL AGENT TAB =====
        with gr.Tab("Email Agent"):
            gr.Markdown("### AI Email Assistant - Chat Interface")

            with gr.Row():
                with gr.Column(scale=1):
                    agent_email_selector = gr.Dropdown(
                        label="Select Email to Analyze",
                        choices=[],
                        interactive=True
                    )
                    load_agent_emails_btn = gr.Button("Load Emails")

                    gr.Markdown("### Quick Actions")
                    summarize_btn = gr.Button("Summarize")
                    extract_actions_btn = gr.Button("Extract Actions")
                    categorize_btn = gr.Button("Categorize")
                    draft_reply_btn = gr.Button("Draft Reply")

                with gr.Column(scale=2):
                    agent_email_preview = gr.Textbox(
                        label="Email Preview",
                        lines=6,
                        interactive=False
                    )

            # Chat interface
            with gr.Row():
                user_query = gr.Textbox(
                    label="Ask the Email Agent",
                    placeholder="e.g., What are the main action items? Summarize this email. Draft a professional reply...",
                    lines=2
                )

            with gr.Row():
                submit_query_btn = gr.Button("Ask Agent", variant="primary")
                clear_chat_btn = gr.Button("Clear")

            with gr.Row():
                agent_response = gr.Textbox(
                    label="Agent Response",
                    lines=8,
                    interactive=False
                )

        # ===== PROMPT BRAIN TAB =====
        with gr.Tab("Prompt Brain"):
            gr.Markdown("### Prompt Configuration & Management")

            with gr.Row():
                with gr.Column(scale=1):
                    prompt_selector = gr.Dropdown(
                        choices=["categorization", "action_extraction", "auto_reply_draft", "summary_generation", "email_agent_general"],
                        label="Select Prompt Template",
                        interactive=True
                    )
                    load_prompt_btn = gr.Button("Load Prompt", variant="primary")

                    gr.Markdown("### Prompt Info")
                    prompt_description = gr.Textbox(label="Description", interactive=False)
                    prompt_category = gr.Textbox(label="Category", interactive=False)

                with gr.Column(scale=2):
                    prompt_content = gr.Textbox(
                        label="Prompt Content",
                        lines=12,
                        interactive=True
                    )

            with gr.Row():
                save_prompt_btn = gr.Button("Save Changes", variant="primary")
                reset_prompt_btn = gr.Button("Reset to Default")
                prompt_status = gr.Textbox(label="Status", interactive=False)

        # ===== DRAFT COMPOSER TAB =====
        with gr.Tab("Draft Composer"):
            gr.Markdown("### Email Draft Generation & Management")

            with gr.Tab("Create New Draft"):
                with gr.Row():
                    draft_email_selector = gr.Dropdown(
                        label="Select Email to Reply To",
                        choices=[],
                        interactive=True
                    )
                    load_draft_emails_btn = gr.Button("Load Emails")

                with gr.Row():
                    draft_subject = gr.Textbox(
                        label="Subject",
                        placeholder="Re: Original Subject...",
                        interactive=True
                    )

                with gr.Row():
                    draft_body = gr.Textbox(
                        label="Draft Body",
                        lines=10,
                        placeholder="Compose your email here...",
                        interactive=True
                    )

                with gr.Row():
                    generate_draft_btn = gr.Button("AI Generated Draft", variant="primary")
                    save_draft_btn = gr.Button("Save Draft")
                    clear_draft_btn = gr.Button("Clear")

                with gr.Row():
                    draft_status = gr.Textbox(label="Status", interactive=False)

            with gr.Tab("View Saved Drafts"):
                gr.Markdown("### Saved Drafts")

                with gr.Row():
                    load_drafts_btn = gr.Button("Refresh Drafts List", variant="primary")
                    delete_draft_btn = gr.Button("Delete Selected Draft", variant="secondary")
                    clear_content_btn = gr.Button("Clear Draft Content", variant="secondary")

                with gr.Row():
                    drafts_display = gr.Dataframe(
                        label="Saved Drafts",
                        headers=["Original Sender", "Original Subject", "Draft Subject", "Last Updated"],
                        datatype=["str", "str", "str", "str"],
                        row_count=5,
                        col_count=(4, "fixed"),
                        interactive=True,
                        type="array"
                    )

                with gr.Row():
                    selected_draft_info = gr.Textbox(
                        label="Draft Information",
                        lines=5,
                        interactive=False
                    )

                with gr.Row():
                    edit_draft_subject = gr.Textbox(
                        label="Draft Subject",
                        interactive=True
                    )

                with gr.Row():
                    edit_draft_body = gr.Textbox(
                        label="Draft Body",
                        lines=8,
                        interactive=True
                    )

                with gr.Row():
                    update_draft_btn = gr.Button("Update Draft", variant="primary")
                    edit_draft_status = gr.Textbox(label="Status", interactive=False)

        # ===== EVENT HANDLERS =====

        def load_inbox_data(category_filter="All"):
            """Load inbox data"""
            conn = sqlite3.connect(DATABASE_NAME)
            if category_filter == "All":
                query = '''SELECT sender, subject, timestamp, category, summary
                          FROM emails ORDER BY timestamp DESC'''
                emails_df = pd.read_sql(query, conn)
            else:
                query = '''SELECT sender, subject, timestamp, category, summary
                          FROM emails WHERE category = ? ORDER BY timestamp DESC'''
                emails_df = pd.read_sql(query, conn, params=(category_filter,))

            # Get stats
            stats_query = '''SELECT category, COUNT(*) as count FROM emails
                           GROUP BY category'''
            stats_df = pd.read_sql(stats_query, conn)
            stats_text = ""
            for _, row in stats_df.iterrows():
                stats_text += f"{row['category']}: {row['count']}\n"

            total_emails = len(emails_df)
            processed = len(emails_df[emails_df['category'].notna()])
            stats_text += f"\nTotal: {total_emails} | Processed: {processed}"

            conn.close()
            return emails_df, stats_text

        def load_email_choices():
            """Load email choices for dropdowns"""
            conn = sqlite3.connect(DATABASE_NAME)
            emails_df = pd.read_sql('SELECT id, sender, subject FROM emails ORDER BY timestamp DESC', conn)
            conn.close()

            choices = [f"{row['sender']} - {row['subject']}" for _, row in emails_df.iterrows()]
            email_ids = emails_df['id'].tolist()

            return {
                email_selector: gr.update(choices=choices),
                agent_email_selector: gr.update(choices=choices),
                draft_email_selector: gr.update(choices=choices),
                email_choices_state: choices,
                email_ids_state: email_ids
            }

        def get_email_details(email_choice, email_choices, email_ids):
            """Get details for selected email"""
            if not email_choice or not email_choices:
                return "", "", "", ""

            idx = email_choices.index(email_choice)
            email_id = email_ids[idx]

            conn = sqlite3.connect(DATABASE_NAME)
            cursor = conn.cursor()
            cursor.execute('SELECT sender, subject, body, category, summary FROM emails WHERE id = ?', (email_id,))
            email_data = cursor.fetchone()
            conn.close()

            if email_data:
                sender, subject, body, category, summary = email_data
                info = f"From: {sender}\nSubject: {subject}\nCategory: {category or 'Not processed'}\nSummary: {summary or 'Not processed'}"
                return info, body, category or "Not processed", summary or "Not processed"
            return "", "", "", ""

        def get_email_preview_only(email_choice, email_choices, email_ids):
            """Get email preview without category for agent tab only"""
            if not email_choice or not email_choices:
                return ""

            idx = email_choices.index(email_choice)
            email_id = email_ids[idx]

            conn = sqlite3.connect(DATABASE_NAME)
            cursor = conn.cursor()
            cursor.execute('SELECT sender, subject, body FROM emails WHERE id = ?', (email_id,))
            email_data = cursor.fetchone()
            conn.close()

            if email_data:
                sender, subject, body = email_data
                preview = f"From: {sender}\nSubject: {subject}\n\nBody:\n{body}"
                return preview

            return ""

        def process_selected_email(email_choice, email_choices, email_ids):
            """Process selected email"""
            if not email_choice or not email_choices:
                return "Select an email", "Select an email"

            idx = email_choices.index(email_choice)
            email_id = email_ids[idx]

            result = process_single_email(email_id)
            return result.get('category', ''), result.get('summary', '')

        def chat_with_selected_email(email_choice, email_choices, email_ids, user_query):
            """Chat with selected email"""
            if not email_choice or not email_choices:
                return "Please select an email first."
            if not user_query:
                return "Please enter a question."

            idx = email_choices.index(email_choice)
            email_id = email_ids[idx]

            return chat_with_email_agent(email_id, user_query)

        def generate_reply_for_email(email_choice, email_choices, email_ids):
            """Generate reply for selected email"""
            if not email_choice or not email_choices:
                return "Please select an email first."

            idx = email_choices.index(email_choice)
            email_id = email_ids[idx]

            ai_reply = generate_smart_reply(email_id)

            subject = "Re: Original Subject"
            body = ai_reply

            if "Subject:" in ai_reply:
                lines = ai_reply.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('Subject:'):
                        subject = line.replace('Subject:', '').strip()
                        body = '\n'.join(lines[i+1:]).strip()
                        break

            conn = sqlite3.connect(DATABASE_NAME)
            cursor = conn.cursor()
            cursor.execute('SELECT subject FROM emails WHERE id = ?', (email_id,))
            original_subject = cursor.fetchone()
            conn.close()

            if subject == "Re: Original Subject" and original_subject:
                subject = f"Re: {original_subject[0]}"

            return subject, body

        def load_prompt_details(prompt_name):
            """Load prompt details"""
            if not prompt_name:
                return "", "", ""

            conn = sqlite3.connect(DATABASE_NAME)
            cursor = conn.cursor()
            cursor.execute('SELECT content, description, category FROM prompts WHERE name = ?', (prompt_name,))
            result = cursor.fetchone()
            conn.close()

            if result:
                return result[0], result[1], result[2]
            return "", "", ""

        def update_prompt_details(prompt_name, new_content, new_description):
            """Update prompt details"""
            if not prompt_name:
                return "Please select a prompt first."

            conn = sqlite3.connect(DATABASE_NAME)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE prompts
                SET content = ?, description = ?, updated_at = ?
                WHERE name = ?
            ''', (new_content, new_description, datetime.now().isoformat(), prompt_name))
            conn.commit()
            conn.close()

            if hasattr(llm_engine, 'prompt_cache') and prompt_name in llm_engine.prompt_cache:
                del llm_engine.prompt_cache[prompt_name]

            return f"Prompt '{prompt_name}' updated successfully!"

        def reset_prompt_to_default(prompt_name):
            """Reset prompt to default values"""
            if not prompt_name:
                return "Please select a prompt first.", "", "", ""

            default_prompts = {
                "categorization": {
                    "content": """Categorize the following email into exactly one of these categories: Important, Newsletter, Spam, To-Do.

            Email Details:
            From: {sender}
            Subject: {subject}
            Body: {body}

            Rules:
            - To-Do emails must include a direct request requiring user action
            - Newsletter emails typically contain mass distribution content, unsubscribe links
            - Spam emails contain suspicious links, prize notifications, or urgent financial requests
            - Important emails are work-related, personal, or require attention but no immediate action

            Respond ONLY with the category name: Important, Newsletter, Spam, or To-Do.""",
                    "description": "Email categorization into predefined categories",
                    "category": "classification"
                },
                "action_extraction": {
                    "content": """Extract all actionable tasks from the following email. Respond in valid JSON format only:

            Email: {body}

            Required JSON structure:
            {
                "action_items": [
                    {
                        "task": "clear description of what needs to be done",
                        "deadline": "extracted deadline or 'Not specified'",
                        "priority": "High/Medium/Low based on urgency language"
                    }
                ]
            }

            Extract deadlines from phrases like "by Friday", "deadline is", "due tomorrow".
            Determine priority: High for urgent/immediate, Medium for soon, Low for whenever.""",
                    "description": "Action item extraction in JSON format",
                    "category": "extraction"
                },
                "auto_reply_draft": {
                    "content": """Draft a professional email reply based on the original email and context.

            Original Email:
            From: {sender}
            Subject: {subject}
            Body: {body}

            User Context: {user_context}

            Guidelines:
            - Maintain professional tone
            - Address the sender appropriately
            - Respond to key points in the original email
            - Keep it concise (3-5 sentences)
            - Do NOT include placeholders like [Name]
            - If it's a meeting request, ask for an agenda
            - If it's a task request, acknowledge and provide timeline
            - Sign off appropriately

            Draft the reply:""",
                    "description": "Auto-reply drafting for emails",
                    "category": "generation"
                },
                "summary_generation": {
                    "content": """Summarize the following email in 2-3 clear bullet points focusing on key information and requests.

            Email:
            From: {sender}
            Subject: {subject}
            Body: {body}

            Create a concise summary that captures:
            - Main purpose of the email
            - Key information shared
            - Any requests or actions needed
            - Important deadlines or dates

            Format as bullet points:""",
                    "description": "Email summarization in bullet points",
                    "category": "summarization"
                },
                "email_agent_general": {
                    "content": """You are an AI email assistant. Help the user with their email query.

            Email Context:
            From: {sender}
            Subject: {subject}
            Body: {body}

            User Query: {user_query}

            Available Actions:
            - Summarize the email
            - Extract action items
            - Categorize the email
            - Draft replies
            - Answer questions about the content

            Provide helpful, accurate information based on the email content.
            If asked to extract action items, list all tasks with deadlines and priorities.
            If asked to categorize, provide the category with explanation.
            If asked to summarize, provide key points and main purpose.

            Response:""",
                    "description": "General email agent for chat interactions",
                    "category": "assistant"
                }
            }

            if prompt_name not in default_prompts:
                return f"Default prompt for {prompt_name} not found.", "", "", ""

            default_prompt = default_prompts[prompt_name]

            conn = sqlite3.connect(DATABASE_NAME)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE prompts
                SET content = ?, description = ?, category = ?, updated_at = ?
                WHERE name = ?
            ''', (
                default_prompt["content"],
                default_prompt["description"],
                default_prompt["category"],
                datetime.now().isoformat(),
                prompt_name
            ))
            conn.commit()
            conn.close()

            if hasattr(llm_engine, 'prompt_cache') and prompt_name in llm_engine.prompt_cache:
                del llm_engine.prompt_cache[prompt_name]

            return f"Prompt '{prompt_name}' reset to default!", default_prompt["content"], default_prompt["description"], default_prompt["category"]

        # ===== DRAFT MANAGEMENT FUNCTIONS =====
        def save_email_draft(email_choice, email_choices, email_ids, subject, body):
            """Save email draft to database"""
            if not email_choice or not email_choices:
                return "Please select an email first.", gr.update(), gr.update(), gr.update(), gr.update()

            if not subject.strip() or not body.strip():
                return "Subject and body cannot be empty.", gr.update(), gr.update(), gr.update(), gr.update()

            idx = email_choices.index(email_choice)
            email_id = email_ids[idx]

            conn = sqlite3.connect(DATABASE_NAME)
            cursor = conn.cursor()

            cursor.execute('SELECT id FROM drafts WHERE email_id = ?', (email_id,))
            existing_draft = cursor.fetchone()

            draft_id = existing_draft[0] if existing_draft else str(uuid.uuid4())
            now = datetime.now().isoformat()

            if existing_draft:
                cursor.execute('''
                    UPDATE drafts
                    SET subject = ?, body = ?, updated_at = ?
                    WHERE id = ?
                ''', (subject, body, now, draft_id))
                action = "updated"
            else:
                cursor.execute('''
                    INSERT INTO drafts (id, email_id, subject, body, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (draft_id, email_id, subject, body, now, now))
                action = "saved"

            conn.commit()
            conn.close()

            drafts_df = load_saved_drafts()

            if len(drafts_df) > 0:
                draft_info = f"Original From: {drafts_df.iloc[0]['Original Sender']}\nOriginal Subject: {drafts_df.iloc[0]['Original Subject']}\nDraft Subject: {subject}"
                return f"Draft {action} successfully!", drafts_df, draft_info, subject, body
            else:
                return f"Draft {action} successfully!", drafts_df, "", "", ""

        def load_saved_drafts():
            """Load all saved drafts from database"""
            conn = sqlite3.connect(DATABASE_NAME)

            query = '''
                SELECT d.id, e.sender, e.subject as original_subject,
                       d.subject as draft_subject, d.body, d.updated_at
                FROM drafts d
                JOIN emails e ON d.email_id = e.id
                ORDER BY d.updated_at DESC
            '''

            drafts_df = pd.read_sql(query, conn)
            conn.close()

            if len(drafts_df) == 0:
                return pd.DataFrame(columns=["Original Sender", "Original Subject", "Draft Subject", "Last Updated"])

            display_df = pd.DataFrame({
                "Original Sender": drafts_df['sender'],
                "Original Subject": drafts_df['original_subject'],
                "Draft Subject": drafts_df['draft_subject'],
                "Last Updated": pd.to_datetime(drafts_df['updated_at']).dt.strftime('%Y-%m-%d %H:%M')
            })

            return display_df

        def get_draft_details(draft_data):
            """Get details for selected draft"""
            if not draft_data or not isinstance(draft_data, list) or len(draft_data) == 0:
                return "", "", "", ""

            selected_row = draft_data[0]

            if not selected_row or len(selected_row) < 3:
                return "", "", "", ""

            original_sender = selected_row[0]
            original_subject = selected_row[1]
            draft_subject = selected_row[2]

            conn = sqlite3.connect(DATABASE_NAME)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT d.id, d.body
                FROM drafts d
                WHERE d.subject = ?
            ''', (draft_subject,))

            draft_result = cursor.fetchone()
            conn.close()

            if draft_result:
                draft_id, draft_body = draft_result
                info = f"Original From: {original_sender}\nOriginal Subject: {original_subject}\nDraft Subject: {draft_subject}"
                return info, draft_subject, draft_body, draft_id

            return "", "", "", ""

        def update_draft(draft_id, subject, body):
            """Update existing draft"""
            if not draft_id:
                return "No draft selected."

            conn = sqlite3.connect(DATABASE_NAME)
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE drafts
                SET subject = ?, body = ?, updated_at = ?
                WHERE id = ?
            ''', (subject, body, datetime.now().isoformat(), draft_id))

            conn.commit()

            cursor.execute('SELECT id FROM drafts WHERE id = ?', (draft_id,))
            updated = cursor.fetchone()
            conn.close()

            if updated:
                return f"Draft updated successfully!"
            else:
                return "Draft not found."

        def delete_draft(draft_data):
            """Delete selected draft"""
            if not draft_data or not isinstance(draft_data, list) or len(draft_data) == 0:
                return "No draft selected.", pd.DataFrame(columns=["Original Sender", "Original Subject", "Draft Subject", "Last Updated"])

            selected_row = draft_data[0]

            if not selected_row or len(selected_row) < 3:
                return "Invalid draft selection.", load_saved_drafts()

            draft_subject = selected_row[2]

            conn = sqlite3.connect(DATABASE_NAME)
            cursor = conn.cursor()

            cursor.execute('DELETE FROM drafts WHERE subject = ?', (draft_subject,))
            conn.commit()

            deleted_count = cursor.rowcount
            conn.close()

            if deleted_count > 0:
                updated_drafts = load_saved_drafts()
                return f"Draft deleted successfully!", updated_drafts
            else:
                return "Draft not found.", load_saved_drafts()

        def clear_draft_content():
            """Clear draft content fields without deleting from database"""
            return "", "", ""

        def clear_chat():
            """Clear chat inputs and responses"""
            return "", ""

        def clear_draft():
            """Clear draft inputs"""
            return "", "", ""

        refresh_inbox_btn.click(
            fn=lambda category: load_inbox_data(category),
            inputs=[category_filter],
            outputs=[inbox_display, stats_display]
        )

        process_all_btn.click(
            fn=lambda: (process_all_emails(), *load_inbox_data("All")),
            outputs=[gr.Textbox(visible=False), inbox_display, stats_display]
        )

        load_emails_btn.click(
            fn=load_email_choices,
            outputs=[email_selector, agent_email_selector, draft_email_selector, email_choices_state, email_ids_state]
        )

        load_agent_emails_btn.click(
            fn=load_email_choices,
            outputs=[email_selector, agent_email_selector, draft_email_selector, email_choices_state, email_ids_state]
        )

        load_draft_emails_btn.click(
            fn=load_email_choices,
            outputs=[email_selector, agent_email_selector, draft_email_selector, email_choices_state, email_ids_state]
        )

        email_selector.change(
            fn=get_email_details,
            inputs=[email_selector, email_choices_state, email_ids_state],
            outputs=[email_info, email_body, category_output, summary_output]
        )

        agent_email_selector.change(
            fn=lambda email_choice, choices, ids: get_email_preview_only(email_choice, choices, ids),
            inputs=[agent_email_selector, email_choices_state, email_ids_state],
            outputs=[agent_email_preview]
        )

        draft_email_selector.change(
            fn=lambda email_choice, choices, ids: get_email_details(email_choice, choices, ids)[1:2],
            inputs=[draft_email_selector, email_choices_state, email_ids_state],
            outputs=[draft_body]
        )

        process_btn.click(
            fn=process_selected_email,
            inputs=[email_selector, email_choices_state, email_ids_state],
            outputs=[category_output, summary_output]
        )

        submit_query_btn.click(
            fn=chat_with_selected_email,
            inputs=[agent_email_selector, email_choices_state, email_ids_state, user_query],
            outputs=[agent_response]
        )

        generate_draft_btn.click(
            fn=generate_reply_for_email,
            inputs=[draft_email_selector, email_choices_state, email_ids_state],
            outputs=[draft_subject, draft_body]
        )

        draft_reply_btn.click(
            fn=generate_reply_for_email,
            inputs=[agent_email_selector, email_choices_state, email_ids_state],
            outputs=[agent_response]
        )

        load_prompt_btn.click(
            fn=load_prompt_details,
            inputs=[prompt_selector],
            outputs=[prompt_content, prompt_description, prompt_category]
        )

        save_prompt_btn.click(
            fn=update_prompt_details,
            inputs=[prompt_selector, prompt_content, prompt_description],
            outputs=[prompt_status]
        )

        reset_prompt_btn.click(
            fn=reset_prompt_to_default,
            inputs=[prompt_selector],
            outputs=[prompt_status, prompt_content, prompt_description, prompt_category]
        )

        save_draft_btn.click(
            fn=save_email_draft,
            inputs=[draft_email_selector, email_choices_state, email_ids_state, draft_subject, draft_body],
            outputs=[draft_status, drafts_display, selected_draft_info, edit_draft_subject, edit_draft_body]
        )

        load_drafts_btn.click(
            fn=load_saved_drafts,
            outputs=[drafts_display]
        )

        drafts_display.select(
            fn=get_draft_details,
            inputs=[drafts_display],
            outputs=[selected_draft_info, edit_draft_subject, edit_draft_body, current_draft_id_state]
        )

        update_draft_btn.click(
            fn=update_draft,
            inputs=[current_draft_id_state, edit_draft_subject, edit_draft_body],
            outputs=[edit_draft_status]
        )

        delete_draft_btn.click(
            fn=delete_draft,
            inputs=[drafts_display],
            outputs=[edit_draft_status, drafts_display]
        )

        clear_chat_btn.click(
            fn=clear_chat,
            outputs=[user_query, agent_response]
        )

        clear_draft_btn.click(
            fn=clear_draft,
            outputs=[draft_subject, draft_body, draft_status]
        )

        clear_content_btn.click(
            fn=clear_draft_content,
            outputs=[selected_draft_info, edit_draft_subject, edit_draft_body]
        )

        summarize_btn.click(
            fn=lambda email_choice, choices, ids: chat_with_selected_email(email_choice, choices, ids, "Please analyze this email and provide a detailed summary with these key points: main purpose, important information, action items needed, and any deadlines."),
            inputs=[agent_email_selector, email_choices_state, email_ids_state],
            outputs=[agent_response]
        )

        extract_actions_btn.click(
            fn=lambda email_choice, choices, ids: chat_with_selected_email(email_choice, choices, ids, "Analyze this email and extract all action items, tasks, and deadlines. For each action, specify what needs to be done, when it's due, and the priority level. Format your response clearly."),
            inputs=[agent_email_selector, email_choices_state, email_ids_state],
            outputs=[agent_response]
        )

        categorize_btn.click(
            fn=lambda email_choice, choices, ids: chat_with_selected_email(email_choice, choices, ids, "Carefully analyze this email and determine its category: Important, Newsletter, Spam, or To-Do. Provide the category and explain your reasoning based on the email content."),
            inputs=[agent_email_selector, email_choices_state, email_ids_state],
            outputs=[agent_response]
        )

        def initial_load():
            inbox_df, stats = load_inbox_data("All")
            email_updates = load_email_choices()
            drafts_df = load_saved_drafts()
            return inbox_df, stats, email_updates[email_choices_state], email_updates[email_ids_state], drafts_df

        demo.load(
            fn=initial_load,
            outputs=[inbox_display, stats_display, email_choices_state, email_ids_state, drafts_display]
        )

    return demo



print("🎨 Creating Gradio Interface...")
demo = create_complete_gradio_interface()

print("🔐 Setting up secure tunnel...")
public_url = setup_secure_tunnel()

print("🚀 Launching InboXpert...")
try:
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
except OSError:
    try:
        print("Port 7860 busy, trying 7861...")
        demo.launch(share=False, server_name="0.0.0.0", server_port=7861)
    except OSError:
        try:
            print("Port 7861 busy, trying 7862...")
            demo.launch(share=False, server_name="0.0.0.0", server_port=7862)
        except OSError:
            print("Port 7862 busy, trying 7863...")
            demo.launch(share=False, server_name="0.0.0.0", server_port=7863)

