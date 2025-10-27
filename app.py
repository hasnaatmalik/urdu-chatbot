import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter
from typing import List, Dict, Tuple

# Set page config
st.set_page_config(
    page_title="Urdu Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    .message-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .stTextInput > div > div > input {
        direction: rtl;
    }
</style>
""", unsafe_allow_html=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============= MODEL ARCHITECTURE (Copy from notebook) =============

class BPETokenizer:
    """Byte Pair Encoding tokenizer with RTL support for Urdu"""
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocab = []
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using BPE"""
        words = text.split()
        tokens = []
        for word in words:
            word_tokens = ['</w>'] + list(word)
            for pair in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i + 1]) == pair:
                        word_tokens = word_tokens[:i] + [''.join(pair)] + word_tokens[i + 2:]
                    else:
                        i += 1
            tokens.extend(word_tokens)
        return tokens

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        tokens = self.tokenize(text)
        return [self.token_to_id.get(token, self.token_to_id.get('<UNK>', 0)) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = [self.id_to_token.get(idx, '<UNK>') for idx in token_ids]
        text = ''.join(tokens).replace('</w> ', ' ').replace('</w>', ' ')
        return text.strip()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=2, d_ff=1024,
                 num_encoder_layers=2, num_decoder_layers=2, max_len=512,
                 dropout=0.1, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.decoder_pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.output_projection = nn.Linear(d_model, vocab_size)

    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.size()
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.encoder_pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        encoder_output = x

        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.decoder_pos_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        output = self.output_projection(x)
        return output


# ============= HELPER FUNCTIONS =============

def generate_text(model, tokenizer, input_text, max_length=30, temperature=0.8, 
                 pad_idx=0, start_idx=1, end_idx=2, unk_idx=3):
    """Generate text continuation"""
    model.eval()
    
    with torch.no_grad():
        tokens = tokenizer.tokenize(input_text)
        token_ids = [tokenizer.token_to_id.get(t, unk_idx) for t in tokens]
        encoder_ids = token_ids[:50] + [pad_idx] * (50 - len(token_ids))
        encoder_input = torch.tensor([encoder_ids], dtype=torch.long).to(device)
        decoder_input = torch.tensor([[start_idx]], dtype=torch.long).to(device)

        generated_tokens = []
        for _ in range(max_length):
            output = model(encoder_input, decoder_input)
            next_token_logits = output[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == end_idx:
                break

            generated_tokens.append(next_token.item())
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)

        output_tokens = []
        for idx in generated_tokens:
            if idx not in [pad_idx, start_idx, unk_idx]:
                token = tokenizer.id_to_token.get(idx, '<UNK>')
                output_tokens.append(token)

        text = ''.join(output_tokens).replace('</w> ', ' ').replace('</w>', ' ')
        return text.strip()


@st.cache_resource
def load_model():
    """Load the trained model and tokenizer"""
    try:
        checkpoint = torch.load('best_model_allinone.pt', map_location=device, weights_only=False)
        config = checkpoint['config']
        tokenizer = checkpoint['tokenizer']
        
        model = Transformer(
            vocab_size=checkpoint['vocab_size'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            max_len=config['max_len'],
            dropout=config['dropout'],
            pad_idx=0
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, tokenizer, config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


# ============= STREAMLIT APP =============

def main():
    # Header
    st.markdown('<p class="main-header">💬 Urdu Chatbot</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )
        
        max_length = st.slider(
            "Max Response Length",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="Maximum number of tokens to generate"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This is an Urdu conversational chatbot built using a Transformer model "
            "with 2-layer encoder-decoder architecture, trained on Urdu text data."
        )
        
        st.markdown("### Model Info")
        if st.session_state.get('model_loaded', False):
            config = st.session_state.get('config', {})
            st.success("✅ Model loaded successfully!")
            st.write(f"**Vocab Size:** {config.get('vocab_size', 'N/A')}")
            st.write(f"**Embedding Dim:** {config.get('d_model', 'N/A')}")
            st.write(f"**Attention Heads:** {config.get('num_heads', 'N/A')}")
            st.write(f"**Device:** {device}")
        
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Load model
    if 'model_loaded' not in st.session_state:
        with st.spinner("Loading model... Please wait"):
            model, tokenizer, config = load_model()
            if model is not None:
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.config = config
                st.session_state.model_loaded = True
            else:
                st.error("❌ Failed to load model. Please ensure 'best_model_allinone.pt' exists in the root directory.")
                st.stop()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-label">👤 You:</div>
                <div>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <div class="message-label">🤖 Bot:</div>
                <div>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message in Urdu:",
            key="user_input",
            placeholder="یہاں اردو میں لکھیں...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send ➤", use_container_width=True)
    
    # Process user input
    if send_button and user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate response
        with st.spinner("Generating response..."):
            try:
                response = generate_text(
                    st.session_state.model,
                    st.session_state.tokenizer,
                    user_input,
                    max_length=max_length,
                    temperature=temperature,
                    pad_idx=0,
                    start_idx=1,
                    end_idx=2,
                    unk_idx=3
                )
                
                # Add bot response to chat
                st.session_state.messages.append({"role": "bot", "content": response})
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
        
        st.rerun()
    
    # Examples section
    if len(st.session_state.messages) == 0:
        st.markdown("### 💡 Try these examples:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("یہ ایک"):
                st.session_state.messages.append({"role": "user", "content": "یہ ایک"})
                st.rerun()
        
        with col2:
            if st.button("پاکستان میں"):
                st.session_state.messages.append({"role": "user", "content": "پاکستان میں"})
                st.rerun()
        
        with col3:
            if st.button("اچھا"):
                st.session_state.messages.append({"role": "user", "content": "اچھا"})
                st.rerun()


if __name__ == "__main__":
    main()