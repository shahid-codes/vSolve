import model_functions
from utils import args
import streamlit as st



if __name__ == "__main__":

    st.title("vSolve: a Question Answering System")

    args = args()
    model = model_functions.load_model(args.base_model_path,args.tuned_model_path)
    tokenizer = model_functions.get_tokenizer(args.base_model_path)
    
    context = st.text_area(label="Context",value="",placeholder="Enter your context here...",height=200)
    question = st.text_input(label="Question",value="",placeholder="Enter your question here...")

    click = st.button(label='Get Answer')

    if click == True: 
 
        if len(context) == 0:
            st.warning('Context is missing. Context cannot be empty', icon="⚠️")
        if len(question) == 0:
            st.warning('Question is missing. Question cannot be empty', icon="⚠️")


        start_logits,end_logits,input_ids = model_functions.get_prediction(model,question,context,tokenizer)
        result, score = model_functions.postprocess(tokenizer,start_logits,end_logits,input_ids)  
        st.text(f"Answer: {result}")
