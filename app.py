import json
import math
import random
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# FRUITZIM LLM
# Decoder-only Transformer treinado do zero
# ============================================================

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "fruitzim_data"
DATA_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = DATA_DIR / "fruitzim_checkpoint.pt"
VOCAB_PATH = DATA_DIR / "vocab.json"
FEEDBACK_PATH = DATA_DIR / "feedback.json"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BUILTIN_CORPUS = """
Fruitzim é uma pequena maçã neural que está aprendendo português.
Olá! Eu sou o Fruitzim. Gosto de conversar, aprender palavras e descobrir ideias.
Como você está hoje? Podemos conversar sobre ciência, tecnologia, livros e criatividade.
Uma rede neural é um programa matemático formado por parâmetros que são ajustados durante o treinamento.
Um modelo de linguagem aprende padrões em textos e tenta prever qual token vem a seguir.
Um Transformer usa atenção para relacionar diferentes partes de uma sequência.
A atenção causal permite que cada posição consulte apenas os tokens anteriores e o token atual.
O treinamento começa com parâmetros aleatórios. Os exemplos de texto ajudam o modelo a ajustar esses parâmetros.
A função de perda mede o quanto as previsões do modelo diferem dos próximos caracteres esperados.
O otimizador Adam atualiza os parâmetros usando gradientes calculados pela retropropagação.
Python é uma linguagem de programação usada em ciência de dados, automação e inteligência artificial.
NumPy oferece operações eficientes com vetores e matrizes.
PyTorch oferece tensores, diferenciação automática e ferramentas para construir redes neurais.
Streamlit permite criar aplicações web interativas usando Python.
A curiosidade é importante: fazer perguntas, testar hipóteses e conferir resultados ajuda a aprender.
Na ciência, uma hipótese precisa ser comparada com evidências e pode ser revisada quando surgem novos dados.
A matemática descreve relações usando números, símbolos, funções e estruturas.
A física investiga matéria, energia, movimento, espaço e tempo.
A biologia estuda os seres vivos e suas relações com o ambiente.
A história analisa acontecimentos humanos usando fontes e interpretações críticas.
A literatura usa linguagem para criar narrativas, imagens, personagens e ideias.
Uma boa explicação organiza o assunto em etapas e apresenta exemplos claros.
Se uma resposta estiver confusa, podemos reformular a pergunta e tentar novamente.
O aprendizado de máquina não é magia: depende de dados, objetivos, parâmetros e avaliação.
Modelos pequenos têm capacidade limitada e podem repetir frases ou produzir erros.
Uma conversa útil exige contexto, clareza e atenção ao que foi realmente perguntado.
O Fruitzim ainda é pequeno, mas pode melhorar quando recebe mais exemplos de texto variados.
"""

SYSTEM_PROMPT = (
    "Você é Fruitzim, uma pequena maçã assistente. "
    "Responda em português, com clareza, educação e honestidade. "
    "Se não souber, admita. Use o contexto da conversa quando disponível.\n"
)


# ============================================================
# TOKENIZER
# ============================================================

class CharTokenizer:

    def __init__(self, chars=None):
        self.special = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.chars = sorted(set(chars or []))

        self.itos = self.special + self.chars
        self.stoi = {
            char: index
            for index, char in enumerate(self.itos)
        }

        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    @classmethod
    def from_text(cls, text):
        return cls(set(text))

    def encode(self, text, add_bos=False, add_eos=False):

        ids = [
            self.stoi.get(char, self.unk_id)
            for char in text
        ]

        if add_bos:
            ids.insert(0, self.bos_id)

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids):

        output = []

        for index in ids:

            index = int(index)

            if 4 <= index < len(self.itos):
                output.append(self.itos[index])

        return "".join(output)

    def to_dict(self):
        return {"chars": self.chars}

    @classmethod
    def from_dict(cls, data):
        return cls(data["chars"])


# ============================================================
# CAUSAL SELF ATTENTION
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        n_embd,
        n_head,
        dropout,
        block_size
    ):

        super().__init__()

        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(
            n_embd,
            3 * n_embd
        )

        self.proj = nn.Linear(
            n_embd,
            n_embd
        )

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(
            torch.ones(block_size, block_size)
        )

        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, block_size, block_size)
        )

    def forward(self, x):

        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(
            B, T, self.n_head, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B, T, self.n_head, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B, T, self.n_head, self.head_dim
        ).transpose(1, 2)

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        scores = scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        weights = self.attn_drop(
            F.softmax(scores, dim=-1)
        )

        y = weights @ v

        y = y.transpose(
            1, 2
        ).contiguous().view(B, T, C)

        return self.resid_drop(
            self.proj(y)
        )


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        n_embd,
        n_head,
        dropout,
        block_size
    ):

        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)

        self.attention = CausalSelfAttention(
            n_embd,
            n_head,
            dropout,
            block_size
        )

        self.ln2 = nn.LayerNorm(n_embd)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        x = x + self.attention(
            self.ln1(x)
        )

        x = x + self.mlp(
            self.ln2(x)
        )

        return x


# ============================================================
# FRUITZIM GPT
# ============================================================

class FruitzimGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        block_size=128,
        n_embd=64,
        n_head=4,
        n_layer=2,
        dropout=0.1
    ):

        super().__init__()

        self.block_size = block_size

        self.token_emb = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.pos_emb = nn.Embedding(
            block_size,
            n_embd
        )

        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                n_embd,
                n_head,
                dropout,
                block_size
            )
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)

        self.head = nn.Linear(
            n_embd,
            vocab_size,
            bias=False
        )

        self.apply(self._init_weights)

        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            T = idx.shape[1]

        positions = torch.arange(
            T,
            device=idx.device
        ).unsqueeze(0)

        x = self.drop(
            self.token_emb(idx) +
            self.pos_emb(positions)
        )

        x = self.blocks(x)

        logits = self.head(
            self.ln_f(x)
        )

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=240,
        temperature=0.85,
        top_k=30
    ):

        self.eval()

        for _ in range(max_new_tokens):

            context = idx[:, -self.block_size:]

            logits, _ = self(context)

            logits = (
                logits[:, -1, :] /
                max(temperature, 0.01)
            )

            if top_k and top_k < logits.size(-1):

                values, _ = torch.topk(
                    logits,
                    top_k
                )

                logits[
                    logits < values[:, [-1]]
                ] = -float("inf")

            probabilities = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

        return idx


# ============================================================
# MODEL MANAGEMENT
# ============================================================

def build_model(tokenizer, config=None):

    config = config or {}

    model = FruitzimGPT(
        vocab_size=len(tokenizer.itos),
        block_size=config.get("block_size", 128),
        n_embd=config.get("n_embd", 64),
        n_head=config.get("n_head", 4),
        n_layer=config.get("n_layer", 2),
        dropout=config.get("dropout", 0.1)
    )

    return model.to(DEVICE)


def save_checkpoint(
    model,
    tokenizer,
    config,
    trained_steps,
    corpus_chars
):

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "trained_steps": trained_steps,
            "corpus_chars": corpus_chars
        },
        CHECKPOINT_PATH
    )

    VOCAB_PATH.write_text(
        json.dumps(
            tokenizer.to_dict(),
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


def load_checkpoint():

    if not CHECKPOINT_PATH.exists():
        return None

    if not VOCAB_PATH.exists():
        return None

    try:

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False
        )

        tokenizer = CharTokenizer.from_dict(
            json.loads(
                VOCAB_PATH.read_text(
                    encoding="utf-8"
                )
            )
        )

        model = build_model(
            tokenizer,
            checkpoint["config"]
        )

        model.load_state_dict(
            checkpoint["model_state"]
        )

        return model, tokenizer, checkpoint

    except Exception as error:

        st.warning(
            f"Erro ao carregar modelo: {error}"
        )

        return None


# ============================================================
# TRAINING
# ============================================================

def make_batches(data, batch_size, block_size):

    max_start = len(data) - block_size - 1

    if max_start <= 0:
        raise ValueError(
            "Corpus insuficiente para o contexto escolhido."
        )

    starts = torch.randint(
        0,
        max_start,
        (batch_size,)
    )

    x = torch.stack([
        data[start:start + block_size]
        for start in starts
    ])

    y = torch.stack([
        data[start + 1:start + block_size + 1]
        for start in starts
    ])

    return x.to(DEVICE), y.to(DEVICE)


def train_model(
    model,
    tokenizer,
    corpus,
    steps,
    batch_size,
    learning_rate,
    block_size,
    progress=None
):

    ids = tokenizer.encode(corpus)

    if len(ids) < block_size + 2:

        raise ValueError(
            "Adicione mais texto ao corpus."
        )

    data = torch.tensor(
        ids,
        dtype=torch.long
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    model.train()

    losses = []

    for step in range(steps):

        xb, yb = make_batches(
            data,
            batch_size,
            block_size
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        _, loss = model(xb, yb)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        losses.append(
            float(loss.item())
        )

        if progress:

            progress_value = (
                (step + 1) / steps
            )

            progress(
                progress_value,
                f"Etapa {step + 1}/{steps} · "
                f"Perda: {loss.item():.4f}"
            )

    return float(
        np.mean(
            losses[-min(20, len(losses)):]
        )
    )


# ============================================================
# FEEDBACK
# ============================================================

def save_feedback(prompt, response, rating):

    entry = {
        "prompt": prompt,
        "response": response,
        "rating": rating,
        "timestamp": time.time()
    }

    entries = []

    if FEEDBACK_PATH.exists():

        try:

            entries = json.loads(
                FEEDBACK_PATH.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:
            entries = []

    entries.append(entry)

    FEEDBACK_PATH.write_text(
        json.dumps(
            entries[-1000:],
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


# ============================================================
# GENERATION
# ============================================================

def generate_reply(
    prompt,
    history,
    max_new_tokens,
    temperature,
    top_k
):

    model = st.session_state.model
    tokenizer = st.session_state.tokenizer

    conversation = SYSTEM_PROMPT

    for message in history[-6:]:

        role = (
            "Usuário"
            if message["role"] == "user"
            else "Fruitzim"
        )

        conversation += (
            f"{role}: {message['content']}\n"
        )

    conversation += (
        f"Usuário: {prompt}\nFruitzim:"
    )

    ids = tokenizer.encode(
        conversation,
        add_bos=True
    )

    idx = torch.tensor(
        [ids],
        dtype=torch.long,
        device=DEVICE
    )

    output = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )

    generated = tokenizer.decode(
        output[0, len(ids):].tolist()
    )

    generated = generated.split(
        "\nUsuário:"
    )[0]

    generated = generated.split(
        "\nFruitzim:"
    )[0]

    generated = generated.strip()

    if not generated:

        return (
            "Ainda estou aprendendo a formar "
            "respostas. Treine-me com mais textos! 🍎"
        )

    return generated


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session():

    if "model" not in st.session_state:

        loaded = load_checkpoint()

        if loaded:

            (
                st.session_state.model,
                st.session_state.tokenizer,
                checkpoint
            ) = loaded

            st.session_state.trained_steps = (
                checkpoint.get("trained_steps", 0)
            )

            st.session_state.corpus_chars = (
                checkpoint.get("corpus_chars", 0)
            )

            st.session_state.config = (
                checkpoint["config"]
            )

        else:

            st.session_state.model = None
            st.session_state.tokenizer = None
            st.session_state.trained_steps = 0
            st.session_state.corpus_chars = 0
            st.session_state.config = None

    if "messages" not in st.session_state:
        st.session_state.messages = []


# ============================================================
# INTERFACE
# ============================================================

st.set_page_config(
    page_title="Fruitzim LLM",
    page_icon="🍎",
    layout="wide"
)

initialize_session()

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            145deg,
            #fffaf2,
            #f4fff5
        );
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,.8);
        padding: 12px;
        border-radius: 14px;
        border: 1px solid #e7eee5;
    }

    .fruit-card {
        background: white;
        border: 1px solid #e8eee6;
        border-radius: 22px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 8px 28px rgba(30,70,35,.06);
    }

    .fruit-face {
        font-size: 76px;
        line-height: 1.2;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🍎 Fruitzim LLM")

st.caption(
    "Uma pequena maçã neural que aprende "
    "padrões de linguagem e gera texto."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧠 Arquitetura neural")

    st.write(
        f"Dispositivo: **{DEVICE.type.upper()}**"
    )

    n_embd = st.select_slider(
        "Dimensão dos embeddings",
        options=[32, 48, 64, 96, 128],
        value=64
    )

    n_head = st.select_slider(
        "Cabeças de atenção",
        options=[2, 4, 8],
        value=4
    )

    n_layer = st.slider(
        "Blocos Transformer",
        1,
        4,
        2
    )

    block_size = st.select_slider(
        "Contexto",
        options=[64, 96, 128, 192, 256],
        value=128
    )

    dropout = st.slider(
        "Dropout",
        0.0,
        0.4,
        0.1,
        0.05
    )

    st.divider()

    st.header("📚 Corpus de treinamento")

    uploaded_file = st.file_uploader(
        "Adicionar arquivo TXT",
        type=["txt"]
    )

    uploaded_text = ""

    if uploaded_file:

        uploaded_text = uploaded_file.getvalue().decode(
            "utf-8",
            errors="replace"
        )

    include_builtin = st.checkbox(
        "Incluir corpus inicial",
        value=True
    )

    corpus = (
        BUILTIN_CORPUS if include_builtin else ""
    ) + "\n\n" + uploaded_text

    st.caption(
        f"{len(corpus):,} caracteres disponíveis"
    )

    st.divider()

    st.header("⚙️ Treinamento")

    steps = st.slider(
        "Etapas",
        20,
        1000,
        200,
        20
    )

    batch_size = st.select_slider(
        "Batch size",
        options=[8, 16, 24, 32],
        value=16
    )

    learning_rate = st.select_slider(
        "Taxa de aprendizado",
        options=[
            0.0001,
            0.0003,
            0.0005,
            0.001,
            0.003
        ],
        value=0.001,
        format_func=lambda value: f"{value:g}"
    )

    train_button = st.button(
        "🍏 Treinar Fruitzim",
        type="primary",
        use_container_width=True
    )

    reset_button = st.button(
        "♻️ Reinicializar modelo",
        use_container_width=True
    )


# ============================================================
# RESET
# ============================================================

if reset_button:

    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.trained_steps = 0
    st.session_state.corpus_chars = 0
    st.session_state.config = None
    st.session_state.messages = []

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    if VOCAB_PATH.exists():
        VOCAB_PATH.unlink()

    st.rerun()


# ============================================================
# TRAINING EXECUTION
# ============================================================

if train_button:

    if not corpus.strip():

        st.error(
            "Adicione textos para treinar o Fruitzim."
        )

    elif n_embd % n_head != 0:

        st.error(
            "A dimensão precisa ser divisível "
            "pelo número de cabeças."
        )

    else:

        config = {
            "block_size": block_size,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "dropout": dropout
        }

        tokenizer = CharTokenizer.from_text(
            corpus
        )

        old_model = st.session_state.model

        architecture_changed = (
            old_model is None
            or st.session_state.config != config
            or st.session_state.tokenizer.itos
            != tokenizer.itos
        )

        if architecture_changed:

            st.session_state.model = build_model(
                tokenizer,
                config
            )

            st.session_state.tokenizer = tokenizer
            st.session_state.trained_steps = 0
            st.session_state.config = config

        progress_bar = st.progress(0.0)
        status = st.empty()

        try:

            average_loss = train_model(
                st.session_state.model,
                st.session_state.tokenizer,
                corpus,
                steps,
                batch_size,
                learning_rate,
                block_size,
                progress=lambda value, message: (
                    progress_bar.progress(value),
                    status.caption(message)
                )
            )

            st.session_state.trained_steps += steps

            st.session_state.corpus_chars = len(corpus)

            save_checkpoint(
                st.session_state.model,
                st.session_state.tokenizer,
                config,
                st.session_state.trained_steps,
                st.session_state.corpus_chars
            )

            st.success(
                f"Treinamento concluído! "
                f"Perda recente: {average_loss:.4f}"
            )

        except Exception as error:

            st.error(
                f"Erro no treinamento: {error}"
            )


# ============================================================
# DASHBOARD
# ============================================================

model = st.session_state.model

if model is None:

    face = "🥺"
    mood = "Aguardando treinamento"
    parameters = 0
    weight_norm = 0.0

else:

    face = "🍎"
    mood = "Pronto para aprender!"

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    weight_norm = math.sqrt(
        sum(
            float(
                torch.sum(
                    parameter.detach() ** 2
                ).item()
            )
            for parameter in model.parameters()
        )
    )


left, right = st.columns([1, 2])

with left:

    st.markdown(
        f"""
        <div class="fruit-card">
            <div class="fruit-face">{face}</div>
            <h2>Fruitzim</h2>
            <p>{mood}</p>
            <small>Neural Language Model</small>
        </div>
        """,
        unsafe_allow_html=True
    )

with right:

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Parâmetros",
        f"{parameters:,}"
    )

    col2.metric(
        "Etapas de treinamento",
        f"{st.session_state.trained_steps:,}"
    )

    col3.metric(
        "Norma L2",
        f"{weight_norm:,.2f}"
    )

    st.info(
        "O Fruitzim começa com pesos aleatórios. "
        "Seu conhecimento depende dos textos usados "
        "durante o treinamento."
    )


# ============================================================
# CHAT
# ============================================================

st.divider()

st.subheader("💬 Conversa com Fruitzim")

if model is None or st.session_state.trained_steps == 0:

    st.warning(
        "Treine o modelo antes de iniciar uma conversa."
    )

else:

    temperature = st.slider(
        "Criatividade",
        0.2,
        1.5,
        0.8,
        0.05
    )

    top_k = st.slider(
        "Top-k",
        1,
        min(50, len(st.session_state.tokenizer.itos)),
        min(30, len(st.session_state.tokenizer.itos))
    )

    max_new_tokens = st.slider(
        "Caracteres gerados",
        40,
        400,
        180,
        20
    )

    for index, message in enumerate(
        st.session_state.messages
    ):

        with st.chat_message(
            message["role"],
            avatar=(
                "🍎"
                if message["role"] == "assistant"
                else "🙂"
            )
        ):

            st.markdown(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and message.get("prompt")
            ):

                st.caption(
                    "Resposta gerada pelo Transformer"
                )

                feedback_cols = st.columns(3)

                if feedback_cols[0].button(
                    "💚 Boa",
                    key=f"good_{index}"
                ):

                    save_feedback(
                        message["prompt"],
                        message["content"],
                        "positive"
                    )

                    st.toast(
                        "Feedback positivo registrado!"
                    )

                if feedback_cols[1].button(
                    "👎 Ruim",
                    key=f"bad_{index}"
                ):

                    save_feedback(
                        message["prompt"],
                        message["content"],
                        "negative"
                    )

                    st.toast(
                        "Feedback negativo registrado!"
                    )

                if feedback_cols[2].button(
                    "🤍 Neutro",
                    key=f"neutral_{index}"
                ):

                    save_feedback(
                        message["prompt"],
                        message["content"],
                        "neutral"
                    )

                    st.toast(
                        "Feedback neutro registrado!"
                    )

    prompt = st.chat_input(
        "Converse com o Fruitzim..."
    )

    if prompt:

        history = st.session_state.messages.copy()

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message(
            "user",
            avatar="🙂"
        ):

            st.markdown(prompt)

        with st.chat_message(
            "assistant",
            avatar="🍎"
        ):

            with st.spinner(
                "Fruitzim está pensando..."
            ):

                answer = generate_reply(
                    prompt,
                    history,
                    max_new_tokens,
                    temperature,
                    top_k
                )

            st.markdown(answer)

            st.caption(
                "Gerado pelo Transformer do Fruitzim"
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "prompt": prompt
            }
        )

        st.rerun()


# ============================================================
# INFORMATION
# ============================================================

with st.expander(
    "🧠 Como funciona o Fruitzim?"
):

    st.markdown(
        """
        ### Arquitetura

        - Decoder-only Transformer.
        - Token embeddings.
        - Positional embeddings.
        - Causal Self-Attention.
        - Feed Forward Network.
        - GELU.
        - Layer Normalization.
        - Residual Connections.
        - Softmax para previsão do próximo token.

        ### Treinamento

        O Fruitzim aprende a prever o próximo caractere
        em uma sequência textual.

        O treinamento utiliza:

        - Cross Entropy.
        - Backpropagation.
        - AdamW.
        - Gradient Clipping.

        ### Feedback

        As avaliações positivas e negativas são salvas
        em um arquivo JSON.

        Nesta versão, o feedback ainda não modifica
        automaticamente os pesos. Para ensinar o modelo,
        acrescente exemplos revisados ao corpus e execute
        uma nova etapa de treinamento.

        ### Persistência

        Os pesos são salvos em:

        `fruitzim_data/fruitzim_checkpoint.pt`

        O vocabulário é salvo em:

        `fruitzim_data/vocab.json`

        ### Limitações

        Este é um modelo experimental pequeno, treinado
        do zero, sem pesos pré-treinados.

        Ele não possui conhecimento geral prévio.
        Sua qualidade depende da quantidade e da qualidade
        dos textos utilizados no treinamento.
        """
    )

st.caption(
    "🍎 Fruitzim LLM · Modelo experimental "
    "treinado do zero · Python + PyTorch + NumPy"
)
