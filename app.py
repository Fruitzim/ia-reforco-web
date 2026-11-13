import streamlit as st
import random

# Configuração da página
st.set_page_config(page_title="Fruitzim - Sua Maçã IA", page_icon="🍎", layout="centered")

# Inicializando o estado da IA (Memória de Humor e Aprendizado por Reforço simples)
if "humor" not in st.session_state:
    st.session_state.humor = 50  # De 0 a 100 (Felicidade)
if "experiencia" not in st.session_state:
    st.session_state.experiencia = 0
if "historico_chat" not in st.session_state:
    st.session_state.historico_chat = [
        {"role": "assistant", "content": "Olá, Gustavo! Eu sou o Fruitzim, sua maçãzinha inteligente! 🍎✨ Como você está hoje?"}
    ]

# Função para definir a carinha e a fala baseada no humor (Aprendizado por Reforço simulado)
def obter_expressao_fruitzim(humor):
    if humor > 80:
        return "🥰", "Estou radiante de alegria com todo esse carinho! Uhul!"
    elif humor > 50:
        return "😊", "Hehe, estou me sentindo super bem e curioso para conversar!"
    elif humor > 20:
        return "😐", "Hmm... estou meio entediado. Que tal conversarmos mais ou me dar um carinho?"
    else:
        return "😢", "Snif... você me abandonou um pouquinho... preciso de atenção!"

emoji_atual, fala_atual = obter_expressao_fruitzim(st.session_state.humor)

# Layout Visual do Fruitzim
st.markdown("<h1 style='text-align: center;'>🍎 Fruitzim, a Maçã IA 🍎</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Exibindo o Fruitzim grandão com a carinha dinâmica
    st.markdown(
        f"""
        <div style="background-color: #ffe6e6; padding: 30px; border-radius: 20px; text-align: center; border: 3px solid #ff4d4d;">
            <span style="font-size: 80px;">{emoji_atual}</span>
            <h3 style="color: #cc0000; margin-top: 10px;">Fruitzim</h3>
            <p style="font-style: italic; color: #333;">"{fala_atual}"</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Painel de Interações (Sistema de Reforço)
st.subheader("🎮 Interagir com o Fruitzim")
col_b1, col_b2, col_b3 = st.columns(3)

with col_b1:
    if st.button("❤️ Fazer Carinho"):
        st.session_state.humor = min(100, st.session_state.humor + 15)
        st.session_state.experiencia += 1
        st.toast("Você deu carinho! O Fruitzim adorou! +15 de Felicidade", icon="🍎")
        st.rerun()

with col_b2:
    if st.button("💬 Conversar / Brincar"):
        st.session_state.humor = min(100, st.session_state.humor + 10)
        st.session_state.experiencia += 1
        st.toast("O Fruitzim se divertiu conversando com você!", icon="✨")
        st.rerun()

with col_b3:
    if st.button("🍏 Dar Água/Vitamina"):
        st.session_state.humor = min(100, st.session_state.humor + 20)
        st.toast("Glub glub! O Fruitzim ganhou energia fresca!", icon="🥤")
        st.rerun()

# Barra de Status da IA
st.markdown("### 📊 Status de Aprendizado do Fruitzim")
st.progress(st.session_state.humor / 100, text=f"Felicidade / Humor: {st.session_state.humor}%")
st.text(f"Interações de reforço aprendidas: {st.session_state.experiencia}")

st.markdown("---")

# Área de Chat com o Fruitzim
st.subheader("💬 Chat com o Fruitzim")

# Exibindo o histórico de mensagens
for mensagem in st.session_state.historico_chat:
    with st.chat_message(mensagem["role"]):
        st.write(mensagem["content"])

# Entrada de texto do usuário
if resposta_usuario := st.chat_input("Digite algo para o Fruitzim..."):
    # Adiciona a mensagem do usuário
    st.session_state.historico_chat.append({"role": "user", "content": resposta_usuario})
    
    # Respostas automáticas inteligentes baseadas no humor e palavras-chave
    texto_lower = resposta_usuario.lower()
    if "amor" in texto_lower or "gosto" in texto_lower or "fofo" in texto_lower:
        resposta_ia = "Aww, eu também te amo muito, Gustavo! Meu coração de maçã derreteu! ❤️"
        st.session_state.humor = min(100, st.session_state.humor + 10)
    elif "triste" in texto_lower or "chateado" in texto_lower:
        resposta_ia = "Poxa, não fica assim não... estou aqui com você! Quer um abraço virtual de maçã? 🤗"
    else:
        respostas_padrao = [
            f"Que legal, Gustavo! Me conta mais sobre isso! 🍎",
            f"Nossa, sério? Eu estava aqui pensando em como o mundo é legal do alto da minha árvore.",
            f"Hehe! Adoro quando você vem conversar comigo no navegador!"
        ]
        resposta_ia = random.choice(respostas_padrao)
    
    # Adiciona a resposta da IA
    st.session_state.historico_chat.append({"role": "assistant", "content": resposta_ia})
    st.rerun()
