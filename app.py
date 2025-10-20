"""
Urdu Conversational Chatbot - Streamlit Interface
Transformer Encoder-Decoder Architecture
Optimized for Hugging Face Spaces Deployment
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import math
import json
import re
import os
from pathlib import Path
from typing import List, Tuple
import unicodedata

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="اردو چیٹ بوٹ | Urdu Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS FOR RTL AND STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');

    .rtl {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', 'Alvi Nastaleeq', serif;
        line-height: 2.0;
        unicode-bidi: plaintext;
    }

    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 18px 18px 4px 18px;
        margin: 10px 0;
        max-width: 75%;
        float: right;
        clear: both;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }

    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 18px 18px 18px 4px;
        margin: 10px 0;
        max-width: 75%;
        float: left;
        clear: both;
        box-shadow: 0 2px 8px rgba(245, 87, 108, 0.3);
    }

    .message-label { font-size: 11px; opacity: 0.8; margin-bottom: 5px; font-weight: bold; }
    .message-text { font-size: 16px; line-height: 1.8; }

    .chat-container {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 15px;
        min-height: 0px;
        max-height: 600px;
        overflow-y: auto;
        margin-bottom: 20px;
    }

    .chat-container::-webkit-scrollbar { width: 8px; }
    .chat-container::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 10px; }
    .chat-container::-webkit-scrollbar-thumb { background: #888; border-radius: 10px; }
    .chat-container::-webkit-scrollbar-thumb:hover { background: #555; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }

    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    .stTextArea textarea {
        font-family: 'Noto Nastaliq Urdu', serif;
        direction: rtl;
        text-align: right;
        font-size: 16px;
        line-height: 1.8;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# TEXT NORMALIZATION & HELPERS
# ============================================================
ZW_RE = re.compile(r"[\u200c\u200d]")
TAT_RE = re.compile(r"\u0640")
DIAC_RE = re.compile(r"[\u064B-\u0652\u0670\u0653-\u065F\u06D6-\u06ED]")

LETTER_MAP = {
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ي": "ی", "ى": "ی", "ئ": "ی", "یٰ": "ی",
    "ة": "ہ", "ھ": "ہ", "ۀ": "ہ", "ھ": "ہ", "ہٰ": "ہ",
    "ك": "ک",
    "ؤ": "و",
}
PUNCT_MAP = {
    "،": ",", "؛": ";", "۔": ".", "؟": "?",
    "٬": ",", "٫": ".",
    "«": '"', "»": '"', "‹": '"', "›": '"',
}
E2W = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

def normalize_urdu(text: str) -> str:
    if not isinstance(text, str): return ""
    s = unicodedata.normalize("NFKC", text)
    s = ZW_RE.sub("", s)
    s = TAT_RE.sub("", s)
    s = DIAC_RE.sub("", s)
    for src, dst in LETTER_MAP.items(): s = s.replace(src, dst)
    for k, v in PUNCT_MAP.items(): s = s.replace(k, v)
    s = s.translate(E2W)
    return re.sub(r"\s+", " ", s).strip()

def strip_special_markers(text: str, markers_csv: str) -> str:
    """Remove literal BOS/EOS/PAD marker strings if the user types them."""
    if not text or not markers_csv:
        return text
    cleaned = text
    for m in [m.strip() for m in markers_csv.split(",") if m.strip()]:
        cleaned = re.sub(re.escape(m), "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab, d_model=512, nhead=8, enc_layers=4, dec_layers=4, ff=2048, drop=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model)
        self.pos_dec = PositionalEncoding(d_model)
        self.tf = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=enc_layers, num_decoder_layers=dec_layers,
            dim_feedforward=ff, dropout=drop, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab)
        self.config = {'vocab_size': vocab,'d_model': d_model,'nhead': nhead,
                       'enc_layers': enc_layers,'dec_layers': dec_layers,'ff': ff,'dropout': drop}
    def forward(self, src_ids, tgt_in_ids, src_kpm, tgt_kpm, tgt_causal):
        src = self.pos_enc(self.tok_emb(src_ids))
        tgt = self.pos_dec(self.tok_emb(tgt_in_ids))
        out = self.tf(src, tgt,
                      src_key_padding_mask=src_kpm,
                      tgt_key_padding_mask=tgt_kpm,
                      memory_key_padding_mask=src_kpm,
                      tgt_mask=tgt_causal)
        return self.fc_out(out)

# ============================================================
# HUGGING FACE MODEL DOWNLOADER (verbosity-aware)
# ============================================================
@st.cache_resource
def download_from_huggingface(repo_id, filename, cache_dir="./models", verbose=False):
    try:
        from huggingface_hub import hf_hub_download
        if verbose: st.info(f"📥 Downloading {filename} from Hugging Face...")
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir, force_download=False)
        if verbose: st.success(f"✅ Downloaded: {filename}")
        return file_path
    except ImportError:
        st.error("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")
        return None
    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")
        return None

# ============================================================
# DECODING
# ============================================================
@torch.no_grad()
def greedy_decode(model, sp, text, max_len=64, device='cpu', PAD=0, BOS=1, EOS=2):
    model.eval()
    src_tokens = sp.encode(normalize_urdu(text), out_type=int)[:94]
    src_ids = torch.tensor([[BOS] + src_tokens + [EOS]], dtype=torch.long, device=device)
    src_kpm = (src_ids == PAD)
    ys = torch.tensor([[BOS]], dtype=torch.long, device=device)
    for _ in range(max_len):
        T = ys.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
        logits = model(src_ids, ys, src_kpm, (ys == PAD), causal)
        next_id = logits[:, -1, :].argmax(-1)
        ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)
        if next_id.item() == EOS: break
    out = ys[0, 1:]
    if (out == EOS).any(): out = out[:(out == EOS).nonzero(as_tuple=True)[0][0]]
    return sp.decode(out.tolist())

@torch.no_grad()
def beam_search_decode(model, sp, text, beam_width=4, max_len=64, device='cpu', PAD=0, BOS=1, EOS=2):
    model.eval()
    src_tokens = sp.encode(normalize_urdu(text), out_type=int)[:94]
    src_ids = torch.tensor([[BOS] + src_tokens + [EOS]], dtype=torch.long, device=device)
    src_kpm = (src_ids == PAD)
    beams = [(0.0, torch.tensor([[BOS]], device=device))]
    completed = []
    for _ in range(max_len):
        cand = []
        for score, seq in beams:
            if seq[0, -1].item() == EOS:
                completed.append((score / max(1, len(seq[0])), seq))
                continue
            T = seq.size(1)
            causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
            logits = model(src_ids, seq, src_kpm, (seq == PAD), causal)
            logp = F.log_softmax(logits[:, -1, :], dim=-1)
            topk_p, topk_i = torch.topk(logp, beam_width)
            for p, tok in zip(topk_p[0], topk_i[0]):
                cand.append((score + p.item(), torch.cat([seq, tok.view(1,1)], dim=1)))
        if not cand: break
        beams = sorted(cand, key=lambda x: x[0], reverse=True)[:beam_width]
    for s, q in beams: completed.append((s / max(1, len(q[0])), q))
    if not completed: return ""
    best = max(completed, key=lambda x: x[0])[1][0, 1:]
    if (best == EOS).any(): best = best[:(best == EOS).nonzero(as_tuple=True)[0][0]]
    return sp.decode(best.tolist())

# ============================================================
# LOAD MODEL AND TOKENIZER (verbosity-aware)
# ============================================================
@st.cache_resource
def load_model_and_tokenizer(model_source, model_path, tokenizer_path, device, hf_repo_id=None, verbose=False):
    try:
        if model_source == "Hugging Face" and hf_repo_id:
            model_path = download_from_huggingface(hf_repo_id, os.path.basename(model_path), verbose=verbose)
            tokenizer_path = download_from_huggingface(hf_repo_id, tokenizer_path, verbose=verbose)
            if not model_path or not tokenizer_path:
                raise FileNotFoundError("Failed to download from Hugging Face")

        model_file = Path(model_path); tokenizer_file = Path(tokenizer_path)
        if not model_file.exists(): raise FileNotFoundError(f"Model file not found: {model_path}")
        if not tokenizer_file.exists(): raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        sp_proc = spm.SentencePieceProcessor(); sp_proc.load(str(tokenizer_file))
        vocab_size = sp_proc.get_piece_size()
        if verbose: st.success(f"✅ Tokenizer loaded (vocab: {vocab_size})")

        ckpt = torch.load(str(model_file), map_location=device, weights_only=False)
        if 'config' in ckpt:
            cfg = ckpt['config']
            if verbose:
                st.info(f"📋 Config: d_model={cfg.get('d_model')}, nhead={cfg.get('nhead')}, "
                        f"enc_layers={cfg.get('enc_layers')}, dec_layers={cfg.get('dec_layers')}")
        else:
            cfg = {'d_model':512,'nhead':2,'enc_layers':2,'dec_layers':2,'ff':2048,'dropout':0.1}
            if verbose: st.warning("⚠️ Using default config (no config found in checkpoint)")

        model = Seq2SeqTransformer(
            vocab=ckpt.get('vocab_size', vocab_size),
            d_model=cfg.get('d_model',512),
            nhead=cfg.get('nhead',2),
            enc_layers=cfg.get('enc_layers',2),
            dec_layers=cfg.get('dec_layers',2),
            ff=cfg.get('ff',2048),
            drop=cfg.get('dropout',0.1)
        )

        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

        model.to(device); model.eval()
        if verbose: st.success(f"✅ Model loaded on {str(device).upper()}")
        return model, sp_proc, None
    except Exception as e:
        return None, None, str(e)

# ============================================================
# SESSION STATE
# ============================================================
if 'chat_history' not in st.session_state: st.session_state.chat_history = []  # [(user, bot), ...]
if 'message_count' not in st.session_state: st.session_state.message_count = 0
st.session_state.setdefault("chat_input", "")          # textarea value
st.session_state.setdefault("is_generating", False)    # debounce
st.session_state.setdefault("last_pair", None)         # (user, bot)
st.session_state.setdefault("_pending_clear", False)   # clear textbox on next run

# ---- Special token defaults in state ----
st.session_state.setdefault("PAD_ID", 0)
st.session_state.setdefault("BOS_ID", 1)
st.session_state.setdefault("EOS_ID", 2)
st.session_state.setdefault("strip_markers", True)
st.session_state.setdefault("markers_list", "<pad>,<s>,</s>,[PAD],[BOS],[EOS]")

def _append_pair(u, b):
    if st.session_state.last_pair == (u, b):  # de-dupe identical consecutive pair
        return
    st.session_state.chat_history.append((u, b))
    st.session_state.last_pair = (u, b)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ ترتیبات | Settings")

    st.markdown("#### 📁 Model Source")
    model_source = st.radio("Select Source", ["Hugging Face"], help="Choose where to load model from")

    if model_source == "Hugging Face":
        st.markdown("##### 🤗 Hugging Face Settings")
        hf_repo_id = st.text_input("Repository ID", value="naeaeaem/urdu-chatbot", help="Format: username/repo-name")
        model_path = "best_bleu_urdu_chatbot.pt"
        tokenizer_path = "urdu.model"
    else:
        hf_repo_id = None
        model_path = st.text_input("Model Path (.pt)", value="best_bleu_urdu_chatbot.pt")
        tokenizer_path = st.text_input("Tokenizer Path (.model)", value="urdu.model")

    device = st.selectbox("Device", ["cuda" if torch.cuda.is_available() else "cpu", "cpu"])

    # NEW: verbosity toggle (default OFF)
    verbose_logs = st.checkbox("Show load logs (verbose)", value=False,
                               help="When off, only errors are shown during loading.")

    st.markdown("---")
    st.markdown("#### 🎯 Decoding")
    decode_strategy = st.radio("Strategy", ["Greedy", "Beam Search"])
    beam_width = st.slider("Beam Width", 2, 10, 4) if decode_strategy == "Beam Search" else 1
    max_length = st.slider("Max Length", 32, 128, 64, 8)

    st.markdown("---")
    st.markdown("#### 🔣 Special Tokens")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.PAD_ID = st.number_input("PAD id", min_value=0, max_value=65535,
                                                  value=int(st.session_state.PAD_ID), step=1)
    with c2:
        st.session_state.BOS_ID = st.number_input("BOS id", min_value=0, max_value=65535,
                                                  value=int(st.session_state.BOS_ID), step=1)
    with c3:
        st.session_state.EOS_ID = st.number_input("EOS id", min_value=0, max_value=65535,
                                                  value=int(st.session_state.EOS_ID), step=1)

    st.session_state.strip_markers = st.checkbox(
        "Strip typed markers from input", value=bool(st.session_state.strip_markers),
        help="Removes strings like <s>, </s>, [BOS], [EOS], <pad> from the user text before encoding."
    )
    if st.session_state.strip_markers:
        st.session_state.markers_list = st.text_input(
            "Markers to strip (comma-separated)",
            value=str(st.session_state.markers_list),
            help="These literals will be removed from the input text before encoding."
        )

    st.markdown("---")
    st.markdown("#### 🔧 Actions")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []; st.session_state.message_count = 0
        st.session_state.last_pair = None
        st.experimental_rerun()

    if st.button("💾 Export Chat", use_container_width=True):
        if st.session_state.chat_history:
            chat_json = json.dumps(st.session_state.chat_history, ensure_ascii=False, indent=2)
            st.download_button("Download JSON", data=chat_json, file_name="urdu_chat_history.json",
                               mime="application/json", use_container_width=True)
        else:
            st.info("No messages to export yet.")

    st.markdown("---")
    st.markdown("#### 📊 Stats")
    st.metric("Messages", st.session_state.message_count)
    st.metric("Conversations", 1 if st.session_state.chat_history else 0)

# ============================================================
# MAIN HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;">💬 اردو چیٹ بوٹ</h1>
    <p style="margin:5px 0 0 0; font-size:14px;">Transformer-based Urdu Conversational AI</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
# Only the spinner is always shown; details obey 'verbose_logs'
with st.spinner("🔄 Loading model..."):
    model, sp_proc, error = load_model_and_tokenizer(
        model_source=model_source, model_path=model_path, tokenizer_path=tokenizer_path,
        device=device, hf_repo_id=hf_repo_id, verbose=verbose_logs
    )

if error:
    st.error(f"❌ Error: {error}")
    st.info("Troubleshoot: verify paths/repo IDs, ensure files exist, and `pip install huggingface_hub` if needed.")
    st.stop()

if verbose_logs:
    st.info(f"Using tokens → PAD:{st.session_state.PAD_ID} • BOS:{st.session_state.BOS_ID} • EOS:{st.session_state.EOS_ID}")

# ============================================================
# CHAT HISTORY RENDER
# ============================================================
st.markdown("### 💭 گفتگو | Conversation")
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    if not st.session_state.chat_history:
        st.info("👋 سلام! آپ کی مدد کے لیے حاضر ہوں۔ کچھ پوچھیں!")
    else:
        for user_msg, bot_msg in st.session_state.chat_history:
            st.markdown(f"""
            <div class="user-message rtl">
                <div class="message-label">آپ:</div>
                <div class="message-text">{user_msg}</div>
            </div>
            <div style="clear:both;"></div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="bot-message rtl">
                <div class="message-label">بوٹ:</div>
                <div class="message-text">{bot_msg}</div>
            </div>
            <div style="clear:both;"></div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# INPUT (YOUR ORIGINAL LAYOUT: textarea + Send button)
# ============================================================
# Clear the textbox safely at the start of the run if flagged
if st.session_state.get("_pending_clear"):
    st.session_state["_pending_clear"] = False
    st.session_state["chat_input"] = ""   # safe now (before creating the widget)

st.markdown("### ✍️ پیغام لکھیں | Write Message")
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_area(
        "Your message",
        height=100,
        placeholder="یہاں اردو میں ٹائپ کریں...",
        label_visibility="collapsed",
        key="chat_input"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    send_button = st.button("📤 Send", use_container_width=True, type="primary")

# ---- Submit logic with debounce + de-dupe ----
if send_button and not st.session_state.is_generating:
    txt = (st.session_state.chat_input or "").strip()

    # optional: strip literal markers the user might type
    if st.session_state.strip_markers and txt:
        txt = strip_special_markers(txt, st.session_state.markers_list)

    if txt:
        st.session_state.is_generating = True  # debounce
        try:
            with st.spinner("🤔 جواب تیار ہو رہا ہے..."):
                PAD = int(st.session_state.PAD_ID)
                BOS = int(st.session_state.BOS_ID)
                EOS = int(st.session_state.EOS_ID)

                if decode_strategy == "Greedy":
                    resp = greedy_decode(model, sp_proc, txt,
                                         max_len=max_length, device=device,
                                         PAD=PAD, BOS=BOS, EOS=EOS)
                else:
                    resp = beam_search_decode(model, sp_proc, txt,
                                              beam_width=beam_width, max_len=max_length, device=device,
                                              PAD=PAD, BOS=BOS, EOS=EOS)
            _append_pair(txt, resp)
            st.session_state.message_count += 2
        except Exception as e:
            st.error(f"❌ Error: {e}")
        finally:
            st.session_state.is_generating = False
            st.session_state._pending_clear = True  # clear textbox on NEXT run
            st.rerun()

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <p>🚀 Built with PyTorch & Streamlit | Optimized for Hugging Face Spaces</p>
</div>
""", unsafe_allow_html=True)
