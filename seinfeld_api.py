import streamlit as st
import numpy as np
import pickle


st.image("title_image.png", use_container_width=True)
st.caption("From-scratch LSTM vs fine-tuned GPT-2 — pick a model, a cast, and generate a scene.")


# ---- Hugging Face weights repo ----
HF_REPO = "darieyka/seinfeld-generator-streamlit-weights"



PADDING_MAXLEN = 40  # must match LSTM training

# character lists — casing matches each model's training format
LSTM_MAIN  = ['jerry', 'george', 'kramer', 'elaine']
LSTM_SIDE  = ['morty', 'helen', 'frank', 'susan', 'estelle', 'peterman',
              'puddy', 'jack', 'mickey', 'bania', 'soup_nazi']
LSTM_ROLES = ['woman', 'man', 'doctor', 'clerk', 'waitress', 'guy',
              'manager', 'cop', 'attendant']

GPT2_MAIN  = ['JERRY', 'GEORGE', 'ELAINE', 'KRAMER']
GPT2_SIDE  = ['MORTY', 'HELEN', 'FRANK', 'SUSAN', 'ESTELLE', 'PETERMAN',
              'PUDDY', 'JACK', 'MICKEY', 'BANIA', 'SOUP_NAZI']
GPT2_ROLES = ['WOMAN', 'MAN', 'DOCTOR', 'CLERK', 'WAITRESS', 'GUY',
              'MANAGER', 'COP', 'ATTENDANT']


# ------------------------------------------------------------ loaders

@st.cache_resource
def load_lstm():
    from huggingface_hub import hf_hub_download
    from tensorflow.keras.models import load_model

    keras_path = hf_hub_download(repo_id=HF_REPO, filename="seinfeld_lstm.keras")
    pkl_path   = hf_hub_download(repo_id=HF_REPO, filename="tokenizer.pkl")

    model = load_model(keras_path)
    with open(pkl_path, "rb") as f:
        tokens = pickle.load(f)
    return model, tokens

@st.cache_resource
def load_gpt2():
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(HF_REPO)
    model = GPT2LMHeadModel.from_pretrained(HF_REPO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device


# ------------------------------------------------------------ LSTM generation (mirrors notebook)
def sample(preds, temperature):
    preds = np.asarray(preds).astype('float64')
    preds = np.clip(preds, 1e-10, None)
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.argmax(np.random.multinomial(1, preds, 1))


def generate_conversation_lstm(num_turns, model, tokens, temperature,
                               max_words_per_turn, characters):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    if isinstance(characters, str):
        characters = [characters]
    characters = [c.strip().lower() for c in characters]

    speaker_tags = [f"<{c}>" for c in characters]
    speaker_tags = [t for t in speaker_tags if t in tokens.word_index]
    if not speaker_tags:
        return None

    conversation = ""
    for _ in range(num_turns):
        speaker = np.random.choice(speaker_tags)
        conversation += f"\n{speaker}"
        for _ in range(max_words_per_turn):
            token_list = tokens.texts_to_sequences([conversation])[0]
            token_list = pad_sequences([token_list], maxlen=PADDING_MAXLEN - 1, padding='pre')
            predicted = model.predict(token_list, verbose=0)[0]
            next_index = sample(predicted, temperature)
            next_word = tokens.index_word.get(next_index, "")
            if next_word == '<eos>' or next_word == "":
                break
            conversation += " " + next_word
    return conversation.strip()


def prettify_lstm(raw, characters):
    text = raw
    for c in characters:
        tag = f"<{c.lower()}>"
        name = c.replace('_', ' ').upper()
        text = text.replace(tag, '\n' + name + ':')
    return text.replace('<eos>', '').strip()


# ------------------------------------------------------------ GPT-2 generation (mirrors notebook)
def generate_conversation_gpt2(num_turns, model, tokenizer, device, temperature,
                               max_words_per_turn, characters, history_turns=4):
    if isinstance(characters, str):
        characters = [characters]
    speaker_names = [c.strip().upper() for c in characters]
    if not speaker_names:
        return None

    newline_id = tokenizer.encode("\n")[0]
    turns = []
    for _ in range(num_turns):
        speaker = np.random.choice(speaker_names)
        prompt = "\n".join(turns[-history_turns:] + [f"{speaker}:"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_words_per_turn,
            min_new_tokens=2,
            do_sample=True,
            temperature=temperature,
            top_k=50, top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=newline_id,
        )
        new_text = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        dialogue = new_text.split("\n")[0].strip()
        turns.append(f"{speaker}: {dialogue}")
    return "\n".join(turns)


# ------------------------------------------------------------ UI

model_choice = st.radio("Model", ["LSTM (trained from scratch)", "GPT-2 (fine-tuned)"],
                        horizontal=True)
is_lstm = model_choice.startswith("LSTM")

main_list  = LSTM_MAIN if is_lstm else GPT2_MAIN
side_list  = LSTM_SIDE if is_lstm else GPT2_SIDE
role_list  = LSTM_ROLES if is_lstm else GPT2_ROLES

st.subheader("Cast")
main_pick = st.multiselect("Main characters", main_list, default=main_list[:2])
side_pick = st.multiselect("Side characters", side_list, default=[])
role_pick = st.multiselect("Background roles", role_list, default=[])
selected = main_pick + side_pick + role_pick

st.subheader("Settings")
col1, col2 = st.columns(2)
with col1:
    num_turns = st.slider("Number of turns", 2, 20, 8)
    temperature = st.slider("Temperature", 0.2, 1.5, 0.8, 0.1,
                            help="Lower = safer/repetitive, higher = wilder/less coherent")
with col2:
    max_words = st.slider("Max words per turn", 10, 60, 30)
    if not is_lstm:
        history_turns = st.slider("History turns (GPT-2 context)", 0, 8, 4,
                                  help="How many previous lines GPT-2 sees when writing the next one")

if st.button("Generate scene", type="primary"):
    if not selected:
        st.warning("Pick at least one character.")
    else:
        with st.spinner("Writing the scene..."):
            if is_lstm:
                model, tokens = load_lstm()
                raw = generate_conversation_lstm(num_turns, model, tokens,
                                                 temperature, max_words, selected)
                result = prettify_lstm(raw, selected) if raw else None
            else:
                model, tokenizer, device = load_gpt2()
                result = generate_conversation_gpt2(num_turns, model, tokenizer, device,
                                                    temperature, max_words, selected,
                                                    history_turns)
        if result is None:
            st.error("None of the selected characters are known to this model.")
        else:
            st.markdown("### Scene")
            st.text(result)

st.divider()
st.caption("LSTM: word-level, 40-token context, trained from scratch on Seinfeld scripts. "
           "GPT-2: 124M-parameter transformer pretrained on web text, fine-tuned on the same scripts.")