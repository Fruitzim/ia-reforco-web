from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st


# -----------------------------------------------------------------------------
# Configuração da arquitetura
# -----------------------------------------------------------------------------
# O vetor de entrada contém 64 características numéricas e não é contado como
# camada de neurônios. A contagem solicitada considera as unidades das camadas
# ocultas e da camada de saída: 4096 + 1354 + 8 = 5458.
INPUT_DIM = 64
HASH_DIM = 56
HIDDEN_1 = 4096
HIDDEN_2 = 1354
OUTPUT_DIM = 8
TOTAL_NEURONS = HIDDEN_1 + HIDDEN_2 + OUTPUT_DIM

LEARNING_RATE = 0.0002
MAX_GRAD_NORM = 1.0

ACTIONS: List[str] = [
    "acolhimento",
    "explicação",
    "organização",
    "pergunta",
    "síntese",
    "incentivo",
    "análise",
    "criatividade",
]


# -----------------------------------------------------------------------------
# Vetorização de texto sem dependências externas
# -----------------------------------------------------------------------------
def text_to_vector(text: str) -> np.ndarray:
    """Converte texto em um vetor real de 64 características."""
    normalized = text.casefold()
    tokens = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)

    if not tokens:
        tokens = ["mensagem_vazia"]

    vector = np.zeros(INPUT_DIM, dtype=np.float32)

    # Hashing trick com sinal: palavras e pares de palavras ocupam 56 posições.
    features: List[Tuple[str, float]] = [(token, 1.0) for token in tokens]
    features.extend(
        (f"{tokens[i]}_{tokens[i + 1]}", 0.65)
        for i in range(len(tokens) - 1)
    )

    for feature, weight in features:
        digest = hashlib.blake2b(
            feature.encode("utf-8"), digest_size=8
        ).digest()
        index = int.from_bytes(digest[:4], byteorder="little") % HASH_DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += np.float32(sign * weight)

    # Normalização da parte textual.
    text_norm = float(np.linalg.norm(vector[:HASH_DIM]))
    if text_norm > 0.0:
        vector[:HASH_DIM] /= np.float32(text_norm)

    # Oito características simples adicionais.
    token_set = set(tokens)
    vector[56] = min(len(tokens), 60) / 60.0
    vector[57] = float("?" in text)
    vector[58] = float("!" in text)
    vector[59] = float(bool(token_set & {"não", "nunca", "jamais", "nenhum"}))
    vector[60] = float(bool(token_set & {"ajuda", "socorro", "problema", "dúvida"}))
    vector[61] = float(bool(token_set & {"obrigado", "obrigada", "valeu"}))
    vector[62] = float(text.isupper() and any(char.isalpha() for char in text))
    vector[63] = min(len(text), 1000) / 1000.0

    return vector


# -----------------------------------------------------------------------------
# Rede neural e aprendizado por reforço
# -----------------------------------------------------------------------------
class FruitzimNetwork:
    """MLP ReLU/ReLU/Softmax treinada como contextual bandit via REINFORCE."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

        # Inicialização He para camadas ReLU e inicialização menor na saída.
        self.W1 = self.rng.standard_normal(
            (INPUT_DIM, HIDDEN_1), dtype=np.float32
        ) * np.float32(math.sqrt(2.0 / INPUT_DIM))
        self.b1 = np.zeros(HIDDEN_1, dtype=np.float32)

        self.W2 = self.rng.standard_normal(
            (HIDDEN_1, HIDDEN_2), dtype=np.float32
        ) * np.float32(math.sqrt(2.0 / HIDDEN_1))
        self.b2 = np.zeros(HIDDEN_2, dtype=np.float32)

        self.W3 = self.rng.standard_normal(
            (HIDDEN_2, OUTPUT_DIM), dtype=np.float32
        ) * np.float32(math.sqrt(1.0 / HIDDEN_2))
        self.b3 = np.zeros(OUTPUT_DIM, dtype=np.float32)

        self.training_cycles = 0
        self.reward_baseline = 0.0
        self.has_baseline = False
        self.last_reward: float | None = None
        self.last_active_neurons = 0
        self.last_confidence = 0.0

    def forward(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass: vetor -> ReLU -> ReLU -> Softmax."""
        x = np.asarray(x, dtype=np.float32).reshape(1, INPUT_DIM)

        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)

        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0.0)

        logits = a2 @ self.W3 + self.b3
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(
            exp_logits, axis=1, keepdims=True
        )

        return a1, a2, probabilities

    def choose_action(
        self, text: str
    ) -> Tuple[np.ndarray, int, np.ndarray, int]:
        """Processa a mensagem e sorteia uma ação segundo a política Softmax."""
        x = text_to_vector(text)
        a1, a2, probabilities = self.forward(x)

        probs = probabilities[0].astype(np.float64)
        probs /= probs.sum()
        action = int(self.rng.choice(OUTPUT_DIM, p=probs))

        # Ativo = ReLU positivo nas camadas ocultas ou saída >= probabilidade
        # uniforme. A definição evita contar toda unidade Softmax como ativa.
        active_outputs = int(
            np.count_nonzero(probabilities[0] >= (1.0 / OUTPUT_DIM))
        )
        active_neurons = int(
            np.count_nonzero(a1)
            + np.count_nonzero(a2)
            + active_outputs
        )

        self.last_active_neurons = active_neurons
        self.last_confidence = float(probabilities[0, action])

        return x, action, probabilities[0].copy(), active_neurons

    def reinforce(
        self, x: np.ndarray, action: int, reward: float
    ) -> Tuple[float, float]:
        """Atualiza W1/W2/W3 com gradiente de política REINFORCE.

        Trata cada mensagem como uma rodada de contextual bandit. O objetivo é
        maximizar log(pi(ação | contexto)) * vantagem. O baseline móvel reduz
        a variância da estimativa; o gradiente é limitado antes da atualização.
        """
        if not 0 <= int(action) < OUTPUT_DIM:
            raise ValueError("Ação fora do intervalo válido.")

        x = np.asarray(x, dtype=np.float32).reshape(1, INPUT_DIM)
        a1, a2, probabilities = self.forward(x)

        if self.has_baseline:
            advantage = float(reward - self.reward_baseline)
        else:
            advantage = float(reward)
        advantage = float(np.clip(advantage, -2.0, 2.0))

        # Gradiente da log-probabilidade da ação amostrada.
        grad_logits = -probabilities.copy()
        grad_logits[0, int(action)] += 1.0
        grad_logits *= np.float32(advantage)

        grad_W3 = a2.T @ grad_logits
        grad_b3 = np.sum(grad_logits, axis=0)

        # Calcula os gradientes usando os pesos anteriores à atualização.
        grad_a2 = grad_logits @ self.W3.T
        grad_z2 = grad_a2 * (a2 > 0.0)
        grad_W2 = a1.T @ grad_z2
        grad_b2 = np.sum(grad_z2, axis=0)

        grad_a1 = grad_z2 @ self.W2.T
        grad_z1 = grad_a1 * (a1 > 0.0)
        grad_W1 = x.T @ grad_z1
        grad_b1 = np.sum(grad_z1, axis=0)

        gradients = (
            grad_W1,
            grad_b1,
            grad_W2,
            grad_b2,
            grad_W3,
            grad_b3,
        )
        squared_norm = sum(
            float(np.sum(gradient * gradient, dtype=np.float64))
            for gradient in gradients
        )
        gradient_norm = math.sqrt(squared_norm)
        scale = min(1.0, MAX_GRAD_NORM / (gradient_norm + 1e-12))
        step = np.float32(LEARNING_RATE * scale)

        # Ascensão do gradiente: aumenta a probabilidade de ações recompensadas
        # e diminui a de ações punidas para contextos semelhantes.
        self.W1 += step * grad_W1
        self.b1 += step * grad_b1
        self.W2 += step * grad_W2
        self.b2 += step * grad_b2
        self.W3 += step * grad_W3
        self.b3 += step * grad_b3

        # Baseline móvel da recompensa.
        if self.has_baseline:
            self.reward_baseline = (
                0.9 * self.reward_baseline + 0.1 * float(reward)
            )
        else:
            self.reward_baseline = float(reward)
            self.has_baseline = True

        self.training_cycles += 1
        self.last_reward = float(reward)

        return advantage, gradient_norm

    def weight_l2_norm(self) -> float:
        """Norma L2 conjunta das três matrizes de pesos."""
        squared = (
            float(np.sum(self.W1 * self.W1, dtype=np.float64))
            + float(np.sum(self.W2 * self.W2, dtype=np.float64))
            + float(np.sum(self.W3 * self.W3, dtype=np.float64))
        )
        return math.sqrt(squared)


# -----------------------------------------------------------------------------
# Camada de resposta: a rede escolhe o estilo; os textos são templates locais.
# -----------------------------------------------------------------------------
def make_response(text: str, action: int) -> str:
    """Gera uma resposta simples baseada no estilo escolhido pela rede."""
    excerpt = re.sub(r"\s+", " ", text).strip()
    if len(excerpt) > 150:
        excerpt = excerpt[:147].rstrip() + "..."

    # Escapa caracteres Markdown para tratar a entrada como citação simples.
    excerpt = re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", excerpt)
    topic = excerpt or "sua mensagem"

    responses = [
        (
            f'Entendi o que você trouxe: “{topic}”. '
            "Vou abordar isso com cuidado e separar o que já sabemos "
            "do que ainda precisa ser esclarecido."
        ),
        (
            f'O ponto central parece ser “{topic}”. '
            "Vamos por partes: primeiro definimos o objetivo, depois reunimos "
            "as informações e, por fim, avaliamos as possibilidades."
        ),
        (
            f"Vou organizar a ideia: **{topic}**\n\n"
            "1. Definir o que você quer descobrir.\n"
            "2. Separar fatos de suposições.\n"
            "3. Escolher um próximo passo verificável."
        ),
        (
            f'Quero entender melhor “{topic}”. '
            "Qual é o detalhe dessa situação que mais importa para você?"
        ),
        (
            f'Em resumo, você trouxe “{topic}”. '
            "O próximo passo é transformar isso em uma pergunta ou objetivo "
            "bem definido."
        ),
        (
            f'Você já começou ao colocar “{topic}” em palavras. '
            "Agora vale avançar em uma etapa pequena, concreta e verificável."
        ),
        (
            f'Uma análise inicial de “{topic}” pode considerar três pontos: '
            "contexto, evidências e alternativas. Sem esses elementos, "
            "qualquer conclusão seria precipitada."
        ),
        (
            f'Podemos explorar “{topic}” por outro ângulo: '
            "transformar a ideia em um exemplo, um esquema ou uma pequena "
            "experiência."
        ),
    ]
    return responses[int(action)]


def get_mood(agent: FruitzimNetwork) -> Tuple[str, str]:
    if agent.last_reward is not None:
        if agent.last_reward >= 0.8:
            return "🥰", "Fruitzim está radiante"
        if agent.last_reward > 0.0:
            return "😊", "Fruitzim recebeu carinho"
        if agent.last_reward < 0.0:
            return "🥺", "Fruitzim está recalibrando"

    activation_ratio = agent.last_active_neurons / TOTAL_NEURONS
    if activation_ratio > 0.65:
        return "🤩", "Fruitzim está bem ativado"
    if activation_ratio > 0.0:
        return "🤓", "Fruitzim está concentrado"
    return "🍎", "Fruitzim está aguardando uma mensagem"


# -----------------------------------------------------------------------------
# Estado de sessão e callbacks do Streamlit
# -----------------------------------------------------------------------------
def initialize_session() -> None:
    if "fruitzim_agent" not in st.session_state:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
        st.session_state.fruitzim_agent = FruitzimNetwork(seed=seed)

    if "fruitzim_messages" not in st.session_state:
        st.session_state.fruitzim_messages = []

    if "fruitzim_last_interaction" not in st.session_state:
        st.session_state.fruitzim_last_interaction = None


def apply_feedback(reward: float, label: str) -> None:
    """Callback executado pelos botões de feedback."""
    trace = st.session_state.get("fruitzim_last_interaction")
    if not trace or not trace.get("pending", False):
        return

    agent: FruitzimNetwork = st.session_state.fruitzim_agent
    advantage, gradient_norm = agent.reinforce(
        np.asarray(trace["x"], dtype=np.float32),
        int(trace["action"]),
        float(reward),
    )

    updated_trace = dict(trace)
    updated_trace["pending"] = False
    updated_trace["feedback_label"] = label
    updated_trace["reward"] = float(reward)
    updated_trace["advantage"] = float(advantage)
    updated_trace["gradient_norm"] = float(gradient_norm)
    st.session_state.fruitzim_last_interaction = updated_trace

    messages = st.session_state.fruitzim_messages
    if messages and messages[-1].get("role") == "assistant":
        messages[-1]["feedback"] = label


# -----------------------------------------------------------------------------
# Interface
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fruitzim | Maçã IA",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {max-width: 1100px; padding-top: 1.8rem;}
        .fruit-card {
            border: 1px solid rgba(120, 160, 120, .28);
            border-radius: 22px;
            padding: 20px 16px;
            text-align: center;
            background: linear-gradient(145deg, rgba(235,255,235,.85), rgba(255,248,232,.92));
        }
        .fruit-face {font-size: 5rem; line-height: 1.15;}
        .fruit-name {font-size: 1.55rem; font-weight: 800; margin-top: .3rem;}
        .fruit-mood {font-size: .95rem; opacity: .78; margin-top: .2rem;}
        .small-note {font-size: .86rem; opacity: .78;}
    </style>
    """,
    unsafe_allow_html=True,
)

initialize_session()
agent: FruitzimNetwork = st.session_state.fruitzim_agent

st.title("🍎 Fruitzim")
st.caption("Uma pequena maçã neural: recebe texto, escolhe um estilo de resposta e aprende com seu feedback.")

# O chat_input é processado antes das métricas para que elas reflitam a
# propagação direta da mensagem recém-enviada.
prompt = st.chat_input("Converse com o Fruitzim...")
if prompt:
    previous = st.session_state.fruitzim_last_interaction
    if previous and previous.get("pending", False):
        previous_message_list = st.session_state.fruitzim_messages
        if previous_message_list and previous_message_list[-1].get("role") == "assistant":
            previous_message_list[-1]["feedback"] = "Sem avaliação"

    x, action, probabilities, active_neurons = agent.choose_action(prompt)
    answer = make_response(prompt, action)

    st.session_state.fruitzim_messages.append(
        {"role": "user", "content": prompt}
    )
    st.session_state.fruitzim_messages.append(
        {
            "role": "assistant",
            "content": answer,
            "strategy": ACTIONS[action],
            "confidence": float(probabilities[action]),
        }
    )
    st.session_state.fruitzim_last_interaction = {
        "x": x.tolist(),
        "action": int(action),
        "pending": True,
        "active_neurons": int(active_neurons),
        "confidence": float(probabilities[action]),
        "timestamp": time.time(),
    }

face, mood = get_mood(agent)
weight_norm = agent.weight_l2_norm()
active_display = f"{agent.last_active_neurons:,}".replace(",", ".")
total_display = f"{TOTAL_NEURONS:,}".replace(",", ".")

panel_col, metrics_col = st.columns([1, 2.25], gap="large")
with panel_col:
    st.markdown(
        f"""
        <div class="fruit-card">
            <div class="fruit-face">{face}</div>
            <div class="fruit-name">Fruitzim</div>
            <div class="fruit-mood">{mood}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("O rostinho acompanha a última recompensa e a ativação da rede.")

with metrics_col:
    st.subheader("Painel neural")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Unidades na arquitetura", total_display)
    metric_cols[1].metric("Unidades ativas", active_display)
    metric_cols[2].metric("Ciclos de reforço", f"{agent.training_cycles:,}".replace(",", "."))
    metric_cols[3].metric("Norma L2 dos pesos", f"{weight_norm:.3f}")

    activation_percent = agent.last_active_neurons / TOTAL_NEURONS
    st.caption(
        "Ativação: ReLU positivo nas camadas ocultas + saídas Softmax "
        "com probabilidade igual ou superior à média."
    )
    st.progress(float(np.clip(activation_percent, 0.0, 1.0)))

st.divider()
st.subheader("💬 Conversa")

if not st.session_state.fruitzim_messages:
    st.info("Envie uma mensagem para iniciar. O Fruitzim escolherá uma das oito estratégias de resposta.")

for message in st.session_state.fruitzim_messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.text(message["content"])
    else:
        with st.chat_message("assistant", avatar="🍎"):
            st.markdown(message["content"])
            st.caption(
                f"Estratégia neural: {message.get('strategy', '—')} · "
                f"Probabilidade da ação: {message.get('confidence', 0.0):.1%}"
            )
            if message.get("feedback"):
                st.caption(f"Feedback: {message['feedback']}")

trace = st.session_state.fruitzim_last_interaction
if trace and trace.get("pending", False):
    st.markdown("**Como foi esta resposta? Seu feedback atualiza os pesos da rede.**")
    feedback_cols = st.columns(3)
    feedback_cols[0].button(
        "💚 Carinho (+0,3)",
        use_container_width=True,
        key="fruitzim_feedback_carinho",
        on_click=apply_feedback,
        args=(0.3, "Carinho"),
    )
    feedback_cols[1].button(
        "👍 Boa resposta (+1,0)",
        use_container_width=True,
        key="fruitzim_feedback_boa",
        on_click=apply_feedback,
        args=(1.0, "Boa resposta"),
    )
    feedback_cols[2].button(
        "👎 Punição (-1,0)",
        use_container_width=True,
        key="fruitzim_feedback_punicao",
        on_click=apply_feedback,
        args=(-1.0, "Punição"),
    )
elif trace and trace.get("feedback_label"):
    st.success(
        f"Feedback aplicado: {trace['feedback_label']} · "
        f"vantagem {trace['advantage']:+.3f} · "
        f"norma do gradiente {trace['gradient_norm']:.3f}."
    )

with st.sidebar:
    st.header("🧠 Arquitetura")
    st.code(
        "Entrada: 64 características\n"
        "       ↓\n"
        "Oculta 1: 4.096 · ReLU\n"
        "       ↓\n"
        "Oculta 2: 1.354 · ReLU\n"
        "       ↓\n"
        "Saída: 8 · Softmax\n\n"
        "Total contado: 5.458 unidades",
        language="text",
    )
    st.caption(
        "A entrada é um vetor de características, não uma camada contada na "
        "soma solicitada: 4.096 + 1.354 + 8 = 5.458."
    )

    st.header("🎓 Como o reforço funciona")
    st.write(
        "Cada mensagem é um contexto; a rede escolhe uma estratégia pela "
        "distribuição Softmax. O botão escolhido fornece uma recompensa. "
        "O Fruitzim calcula a vantagem em relação a uma média móvel e aplica "
        "REINFORCE com gradiente limitado em W1, W2, W3 e vieses."
    )
    st.caption(
        "É aprendizado por reforço de uma etapa (contextual bandit), não "
        "treinamento conversacional de longo prazo."
    )

    st.header("ℹ️ Limites do protótipo")
    st.write(
        "A rede é inicializada do zero. Ela aprende a favorecer estilos de "
        "resposta em contextos parecidos, mas não possui conhecimento geral "
        "nem um modelo de linguagem pré-treinado. As frases são templates "
        "locais escolhidos pela saída neural."
    )
    st.write(
        "Os pesos ficam na sessão atual do navegador e na memória do processo. "
        "Eles não são gravados em disco e podem ser perdidos quando a sessão "
        "ou a instância do Streamlit Cloud for reiniciada."
    )

    if st.button("♻️ Reiniciar Fruitzim", use_container_width=True):
        seed = int(np.random.SeedSequence().generate_state(1)[0])
        st.session_state.fruitzim_agent = FruitzimNetwork(seed=seed)
        st.session_state.fruitzim_messages = []
        st.session_state.fruitzim_last_interaction = None
        st.rerun()
