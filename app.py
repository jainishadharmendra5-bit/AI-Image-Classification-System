import streamlit as st
import requests
from fcts import FACTS

st.title("🤖 CIFAR-10 AI Chatbot")


if "data" not in st.session_state:
    st.session_state.data = None


uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:

    res = requests.post(
        "https://ai-image-classification-system.onrender.com/predict",
        files={"file": uploaded_file}
    )

    st.session_state.data = res.json()

    st.success("Image analyzed automatically!")


    st.write("### Prediction:")
    st.write(st.session_state.data["label"])
    st.write(f"{st.session_state.data['confidence']*100:.2f}%")



user_question = st.chat_input("Ask something about the image...")

if user_question:

    if st.session_state.data is None:
        st.warning("Upload an image first.")
    else:

        label = st.session_state.data["label"]
        conf = st.session_state.data["confidence"]


        facts = FACTS.get(label, ["No facts available."])

        q = user_question.lower()

        if "what" in q:
            answer = f"It is a {label}."

        elif "why" in q:
            answer = f"I predicted {label} based on image patterns learned from CIFAR-10."

        elif "fact" in q:
            answer = f"Facts about {label}:\n- " + "\n- ".join(facts)

        elif "confidence" in q:
            answer = f"I am {conf*100:.2f}% confident."

        else:
            answer = f"I think it is a {label}. You can ask about facts, why, or confidence."

        st.chat_message("assistant").write(answer)


