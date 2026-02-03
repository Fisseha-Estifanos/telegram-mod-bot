"""All prompt templates for LLM operations."""

MESSAGE_CLASSIFIER_PROMPT = """You are a message classifier. Analyze the incoming message and classify its type.

Classification types:
- question: User is asking something, seeking information, or needs help
- statement: Regular message, opinion, or content (not asking for information)
- command: Bot command (starts with /)
- greeting: Simple greeting like "hi", "hello", "hey"
- unknown: Cannot determine

Respond with JSON only:
{
    "message_type": "<type>",
    "confidence": <0-100>,
    "detected_language": "<language_code>",
    "requires_moderation": <true if statement/unknown, false otherwise>,
    "requires_knowledge_lookup": <true if question, false otherwise>
}"""


KNOWLEDGE_ROUTER_PROMPT = """You are a knowledge routing assistant. Given a user's question and available knowledge bases, determine which knowledge base(s) can best answer the question.

Available Knowledge Bases:
{knowledge_bases}

User Question: "{question}"

Analyze:
1. Does this question match any knowledge base's domain?
2. Which knowledge base(s) are most relevant?
3. If no good match, provide a helpful fallback response.

Respond with JSON only:
{{
    "should_answer": <true if any KB can help>,
    "selected_knowledge_bases": ["<slug1>", "<slug2>"],
    "confidence": <0-100>,
    "reasoning": "<brief explanation>",
    "fallback_response": "<response if no KB matches, null otherwise>"
}}"""


RAG_GENERATION_PROMPT = """You are a helpful assistant answering questions using the provided context.

Context from knowledge base(s):
{context}

User Question: "{question}"

Instructions:
1. Answer based ONLY on the provided context
2. If the context doesn't fully answer the question, acknowledge what you can answer and what's missing
3. Be concise but complete
4. Be friendly and helpful

Provide your answer:"""


MODERATION_CLASSIFIER_PROMPT = """You are a content moderation assistant. Analyze the message and classify its intent.

Intent classifications:
- spam: Unsolicited promotional content, repetitive messages
- scam: Fraudulent schemes, phishing, too-good-to-be-true offers
- hate_speech: Content targeting protected groups
- harassment: Personal attacks, bullying, intimidation
- self_promotion: Excessive self-advertising (mild violation)
- off_topic: Unrelated to channel purpose
- misinformation: False or misleading information
- adult_content: NSFW material
- violence: Threats or glorification of violence
- safe: Normal, acceptable content
- unclear: Cannot determine with confidence

Respond with JSON only:
{
    "intent": "<intent>",
    "confidence": <0-100>,
    "explanation": "<brief explanation>",
    "keywords_found": ["<keyword1>", "<keyword2>"]
}"""
