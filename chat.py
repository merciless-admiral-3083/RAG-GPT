import torch
import torch.nn.functional as F
import argparse
import tiktoken
import re

from train import GPT, GPTConfig
from rag.rag_retriever import RAGRetriever
from rag.config import (
    BEST_DISTANCE_THRESHOLD,
    TOP_K_RETRIEVAL,
    MAX_CHUNKS_RETURNED,
    MIN_WORD_OVERLAP,
    MAX_CONTEXT_TOKENS,
)

# ----------------------------
# ARGUMENTS
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="log/main/model_chat.pt")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--temperature", type=float, default=0.4)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--max_tokens", type=int, default=100)
parser.add_argument("--rag_weight", type=float, default=0.90, help="RAG weight (0.85-0.95)")
parser.add_argument("--debug", action="store_true", help="Show debug information")
args = parser.parse_args()

DEVICE = args.device
TEMPERATURE = args.temperature
TOP_K = args.top_k
MAX_NEW_TOKENS = args.max_tokens
RAG_WEIGHT = args.rag_weight
DEBUG = args.debug

# ----------------------------
# CONFIGURATION
# ----------------------------
DISTANCE_THRESHOLD = 1.5
MIN_CONTEXT_LENGTH = 5
MIN_KEYWORD_OVERLAP = 0
MAX_RAG_CHUNKS = 5

# ----------------------------
# LOAD MODEL
# ----------------------------
def load_model(path):
    try:
        ckpt = torch.load(path, map_location=DEVICE)
        model = GPT(GPTConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model"])
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        if DEBUG:
            print(f"⚠️  Warning: Could not load GPT model: {e}")
        return None

# ----------------------------
# SMART TEXT CLEANING
# ----------------------------
def clean_text(text):
    """Clean text while preserving important content."""
    if not text:
        return ""
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^[\s•\-*\d.]+$', line):
            continue
        if re.match(r'^\s*[•\-*]?\s*\d{4}\s+(?:ICC|World|Cup|Championship)', line):
            continue
        if line.count('•') > 3:
            continue
        cleaned_lines.append(line)
    
    text = ' '.join(cleaned_lines)
    text = re.sub(r'[•●○]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def is_valid_sentence(sentence):
    """Check if sentence is complete and meaningful."""
    sentence = sentence.strip()
    
    if len(sentence) < 10:
        return False
    
    alpha_count = sum(c.isalpha() for c in sentence)
    if alpha_count < 8:
        return False
    
    # Check for incomplete patterns
    incomplete_patterns = [
        r'\b(?:is|are|was|were)\s+a\s*$',
        r'\b(?:the|an?)\s*$',
        r'\b(?:of|in|on|at|to|for|with|by|from)\s*$',
    ]
    
    for pattern in incomplete_patterns:
        if re.search(pattern, sentence.lower()):
            return False
    
    digit_ratio = sum(c.isdigit() for c in sentence) / max(len(sentence), 1)
    if digit_ratio > 0.6:
        return False
    
    special_chars = '[](){}#@$%^&'
    special_count = sum(c in special_chars for c in sentence)
    if special_count > 7:
        return False
    
    return True

# ----------------------------
# SENTENCE EXTRACTION
# ----------------------------
def extract_sentences(text):
    """Extract valid sentences from text."""
    text = clean_text(text)
    
    if not text:
        return []
    
    # Split on sentence endings while preserving them
    parts = re.split(r'([.!?])\s+', text)
    
    sentences = []
    current = ""
    
    for i, part in enumerate(parts):
        if part.strip() in '.!?':
            current += part
            if current.strip():
                sentences.append(current.strip())
            current = ""
        else:
            current += part
    
    # Add any remaining text
    if current.strip():
        if not current.strip()[-1] in '.!?':
            current += '.'
        sentences.append(current.strip())
    
    # Filter and deduplicate
    valid = []
    seen = set()
    
    for sent in sentences:
        sent = sent.strip()
        
        if not is_valid_sentence(sent):
            continue
        
        normalized = ' '.join(sent.lower().split())
        
        if normalized in seen:
            continue
        
        # Check near-duplicates
        is_dup = False
        for existing in seen:
            words1 = set(normalized.split())
            words2 = set(existing.split())
            if words1 and words2:
                similarity = len(words1 & words2) / max(len(words1), len(words2))
                if similarity > 0.85:
                    if len(normalized.split()) > len(existing.split()):
                        seen.discard(existing)
                        valid = [s for s in valid if ' '.join(s.lower().split()) != existing]
                    else:
                        is_dup = True
                    break
        
        if is_dup:
            continue
        
        seen.add(normalized)
        valid.append(sent)
    
    return valid

# ----------------------------
# ANSWER EXTRACTION
# ----------------------------
def extract_answer(question, context, max_sentences=2):
    """Extract the best answer focusing on the specific question."""
    
    sentences = extract_sentences(context)
    
    if not sentences:
        return None
    
    if DEBUG:
        print(f"\n[DEBUG] Found {len(sentences)} valid sentences")
    
    question_lower = question.lower().strip()
    question_words = set(question_lower.split())
    
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'who', 
        'where', 'when', 'how', 'why', 'which', 'do', 'does', 'did'
    }
    question_keywords = question_words - stop_words
    
    # Detect question type
    is_specific_question = any(question_lower.startswith(phrase) for phrase in [
        'which is', 'what is the', 'who is the', 'when did', 'where is',
        'what is', 'who invented', 'who created'
    ])
    
    scored = []
    
    for sent in sentences:
        score = 0
        sent_lower = sent.lower()
        sent_words = set(sent_lower.split())
        sent_keywords = sent_words - stop_words
        
        # Keyword overlap
        overlap = len(question_keywords & sent_keywords)
        score += overlap * 8
        
        # Question type bonuses
        if question_lower.startswith(('what is', 'what are', 'define')):
            if any(p in sent_lower for p in ['is a', 'is an', 'are', 'refers to', 'means', 'known as', 'consists of']):
                score += 12
        
        elif question_lower.startswith(('who is', 'who was', 'who invented', 'who created')):
            if any(w in sent_lower for w in ['invented', 'created', 'founded', 'published', 'designed']):
                score += 15
            if re.search(r'\b\d{4}\b', sent):
                score += 5
        
        elif question_lower.startswith(('which is', 'what is the most', 'what is the hottest', 'what is the largest')):
            if any(w in sent_lower for w in ['hottest', 'largest', 'biggest', 'most', 'best', 'highest']):
                score += 20
            elif any(w in question_keywords for w in ['hottest', 'largest', 'biggest']) and \
                 not any(w in sent_lower for w in ['hottest', 'largest', 'biggest', 'hotter', 'larger', 'bigger']):
                score -= 10
        
        elif question_lower.startswith(('how', 'how to', 'how do')):
            if any(w in sent_lower for w in ['by', 'through', 'process', 'method', 'using', 'begins']):
                score += 12
        
        elif question_lower.startswith(('when', 'when did', 'when was')):
            if re.search(r'\b\d{4}\b', sent):
                score += 15
            if any(w in sent_lower for w in ['in', 'on', 'during', 'century', 'january', 'february', 'march']):
                score += 8
        
        # Length preference
        word_count = len(sent.split())
        if is_specific_question:
            if 10 <= word_count <= 35:
                score += 6
            elif word_count > 60:
                score -= 5
        else:
            if 15 <= word_count <= 50:
                score += 6
        
        # Position bonus
        pos = sentences.index(sent)
        if pos == 0:
            score += 5
        elif pos == 1:
            score += 3
        
        # Penalize questions
        if '?' in sent:
            score -= 10
        
        scored.append((sent, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    
    if DEBUG:
        print("\n[DEBUG] Top 5 scored sentences:")
        for sent, sc in scored[:min(5, len(scored))]:
            preview = sent[:100] + "..." if len(sent) > 100 else sent
            print(f"  Score {sc}: {preview}")
    
    # For specific questions, prefer single best sentence
    if is_specific_question and scored and scored[0][1] > 12:
        selected = [scored[0][0]]
    else:
        # Take top sentences with good scores
        selected = [s for s, sc in scored if sc > 5][:max_sentences]
    
    if not selected:
        return None
    
    # Build answer - proper sentence joining
    answer_parts = []
    for sent in selected:
        sent = sent.strip()
        # Remove trailing period for joining
        if sent.endswith('.'):
            sent = sent[:-1]
        answer_parts.append(sent)
    
    # Join with period and space
    answer = '. '.join(answer_parts)
    
    # Add final period
    if not answer.endswith(('.', '!', '?')):
        answer += '.'
    
    # Cleanup
    answer = re.sub(r'\s+', ' ', answer).strip()
    answer = re.sub(r'\.\.+', '.', answer)
    
    return answer

# ----------------------------
# CONTEXT VALIDATION
# ----------------------------
def is_acceptable_context(text, question):
    """Validate context quality."""
    if not text or len(text.split()) < MIN_CONTEXT_LENGTH:
        return False
    
    text_lower = text.lower()
    question_lower = question.lower()
    
    code_keywords = ['java', 'python', 'code', 'function', 'program', 'script', 'syntax']
    is_code_q = any(k in question_lower for k in code_keywords)
    has_code = ('{' in text and '}' in text) or 'def ' in text or 'import ' in text
    
    if has_code and not is_code_q:
        return False
    
    bad = ['###', 'step 1:', 'instruction:', 'you must follow']
    if any(b in text_lower for b in bad):
        return False
    
    return True

# ----------------------------
# IMPROVED GPT GENERATION
# ----------------------------
@torch.no_grad()
def gpt_generate(model, enc, context, question):
    """GPT generation with strict repetition control."""
    if model is None:
        return None
    
    context = clean_text(context)
    
    # Concise prompt
    prompt = f"""Context: {context[:350]}

Q: {question}
A:"""
    
    try:
        input_ids = torch.tensor([enc.encode(prompt)], device=DEVICE)
        generated = []
        
        max_gen = min(MAX_NEW_TOKENS, 60)
        
        for step in range(max_gen):
            logits, _ = model(input_ids[:, -model.config.block_size:])
            logits = logits[:, -1, :] / TEMPERATURE
            
            if TOP_K > 0:
                v, _ = torch.topk(logits, TOP_K)
                logits[logits < v[:, [-1]]] = -float("inf")
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check current output every few tokens
            if step > 5 and step % 3 == 0:
                decoded = enc.decode(generated).strip()
                
                # Stop on proper sentence end (after reasonable length)
                if len(decoded) > 25 and decoded.endswith(('.', '!', '?')):
                    break
                
                # Check for repetition - BREAK immediately
                if len(decoded.split()) > 8:
                    words = decoded.lower().split()
                    # Check last 6 words against previous 6 words
                    if len(words) >= 12:
                        last_phrase = ' '.join(words[-6:])
                        prev_phrase = ' '.join(words[-12:-6])
                        if last_phrase == prev_phrase:
                            if DEBUG:
                                print("[DEBUG] Repetition detected, stopping GPT")
                            # Truncate at repetition point
                            generated = generated[:len(generated)//2]
                            break
        
        if not generated:
            return None
        
        answer = enc.decode(generated).strip()
        
        # Clean up answer
        answer = clean_text(answer)
        
        # Remove prompt artifacts
        for prefix in ['answer:', 'a:', 'context:', 'question:', 'q:']:
            if answer.lower().startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Final repetition check
        words = answer.lower().split()
        if len(words) >= 8:
            # Check if same 3-word phrase appears twice
            for i in range(len(words) - 6):
                phrase = ' '.join(words[i:i+3])
                rest = ' '.join(words[i+3:])
                if phrase in rest:
                    # Truncate at first occurrence
                    answer = ' '.join(words[:i+3])
                    break
        
        # Validate minimum quality
        if not answer or len(answer.split()) < 6:
            return None
        
        # Check for garbage output
        alpha_ratio = sum(c.isalpha() for c in answer) / max(len(answer), 1)
        if alpha_ratio < 0.5:
            return None
        
        # Ensure proper ending
        if not answer.endswith(('.', '!', '?')):
            # Find last complete sentence
            for delim in ['.', '!', '?']:
                if delim in answer:
                    answer = answer[:answer.rfind(delim)+1]
                    break
            else:
                answer += '.'
        
        # Final cleanup
        answer = re.sub(r'\s+', ' ', answer).strip()
        answer = re.sub(r'\.\.+', '.', answer)
        
        return answer
        
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] GPT error: {e}")
        return None

# ----------------------------
# HYBRID STRATEGY
# ----------------------------
def get_best_answer(question, context, model, enc):
    """Hybrid RAG-GPT strategy with proper fallback."""
    
    # Try RAG first
    rag_answer = extract_answer(question, context)
    
    # Calculate RAG confidence
    rag_confidence = 0.0
    if rag_answer:
        words = len(rag_answer.split())
        has_numbers = bool(re.search(r'\d', rag_answer))
        question_words = set(question.lower().split())
        answer_words = set(rag_answer.lower().split())
        overlap = len(question_words & answer_words) / max(len(question_words), 1)
        
        # Check if answer matches question type
        question_lower = question.lower()
        answer_lower = rag_answer.lower()
        
        type_match = 1.0
        if 'hottest' in question_lower and 'hot' not in answer_lower:
            type_match = 0.3
        elif 'largest' in question_lower and 'large' not in answer_lower:
            type_match = 0.3
        elif 'who invented' in question_lower and 'invent' not in answer_lower and 'created' not in answer_lower:
            type_match = 0.5
        
        rag_confidence = min(1.0, (words / 25) * 0.3 + overlap * 0.4 + (0.2 if has_numbers else 0) + type_match * 0.1)
    
    if DEBUG:
        print(f"[DEBUG] RAG confidence: {rag_confidence:.2f}")
    
    # High confidence RAG - use it
    if rag_answer and rag_confidence >= RAG_WEIGHT:
        if DEBUG:
            print("[DEBUG] Using RAG answer (high confidence)")
        return rag_answer, 'rag'
    
    # Try GPT only if RAG confidence is low
    gpt_answer = None
    if model is not None and (not rag_answer or rag_confidence < 0.60):
        gpt_answer = gpt_generate(model, enc, context, question)
    
    # Decision logic
    if rag_answer and gpt_answer:
        # Prefer RAG unless confidence is very low
        if rag_confidence < 0.40:
            if DEBUG:
                print("[DEBUG] Using GPT answer (very low RAG confidence)")
            return gpt_answer, 'gpt'
        else:
            if DEBUG:
                print("[DEBUG] Using RAG answer (moderate confidence)")
            return rag_answer, 'rag'
    
    elif gpt_answer:
        if DEBUG:
            print("[DEBUG] Using GPT answer (no good RAG)")
        return gpt_answer, 'gpt'
    
    elif rag_answer:
        if DEBUG:
            print("[DEBUG] Using RAG answer (no GPT)")
        return rag_answer, 'rag'
    
    return None, None

# ----------------------------
# MAIN
# ----------------------------
def main():
    print("=" * 60)
    print("🔒 RAG-GPT Hybrid Chat")
    print("=" * 60)
    
    enc = tiktoken.get_encoding("gpt2")
    model = load_model(args.model)
    
    if model:
        print("✅ GPT model loaded")
    else:
        print("⚠️  GPT model not available - using RAG only")
    
    try:
        rag = RAGRetriever("rag_index/index.faiss", "rag_index/data.json")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return
    
    print("\nType your question (or 'exit' to quit)")
    if DEBUG:
        print("🐛 Debug mode enabled")
    
    stats = {'rag': 0, 'gpt': 0, 'fallback': 0, 'none': 0}
    
    while True:
        print("\n" + "-" * 60)
        q = input("👱 You: ").strip()
        
        if q.lower() in ("exit", "quit", "q"):
                
                
            print("\nGoodbye!👋")
            break
        
        if not q:
            continue
        
        # Retrieve from RAG
        try:
            results = rag.retrieve(q, top_k=TOP_K_RETRIEVAL)
        except Exception as e:
            print(f"\n🤖 Assistant: Error: {e}")
            continue
        
        if not results:
            print("\n🤖 Assistant: I don't have information about that.")
            stats['none'] += 1
            continue
        
        if DEBUG:
            print(f"\n[DEBUG] Retrieved {len(results)}, distance: {results[0]['distance']:.3f}")
        
        # Filter results
        filtered = [r for r in results 
                   if r["distance"] <= DISTANCE_THRESHOLD 
                   and is_acceptable_context(r["text"], q)]
        
        if DEBUG:
            print(f"[DEBUG] Filtered: {len(filtered)}")
        
        if not filtered:
            print("\n🤖 Assistant: I don't have reliable information about that.")
            stats['none'] += 1
            continue
        
        # Build context
        context = ' '.join([r["text"] for r in filtered[:MAX_RAG_CHUNKS]])
        if len(enc.encode(context)) > MAX_CONTEXT_TOKENS:
            context = enc.decode(enc.encode(context)[:MAX_CONTEXT_TOKENS])
        
        if DEBUG:
            print(f"[DEBUG] Context: {len(context.split())} words")
        
        # Get best answer
        answer, source = get_best_answer(q, context, model, enc)
        
        if answer:
            print(f"\n🤖 Assistant: {answer}")
            stats[source] += 1
            continue
        
        # Final fallback to first valid sentence
        if filtered:
            sents = extract_sentences(filtered[0]["text"])
            if sents:
                fb = sents[0]
                if not fb.endswith('.'):
                    fb += '.'
                print(f"\n🤖 Assistant: {fb}")
                stats['fallback'] += 1
                continue
        
        print("\n🤖 Assistant: I don't have reliable information about that.")
        stats['none'] += 1

if __name__ == "__main__":
    main()