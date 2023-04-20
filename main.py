import model_functions
from utils import args
import streamlit as st




if __name__ == "__main__":

    args = args()
    model = model_functions.load_model(args.base_model_path,args.tuned_model_path)
    tokenizer = model_functions.get_tokenizer(args.base_model_path)
    
    home,ex1,ex2 = st.tabs(['Home','Example1','Example2'])

    with home:
        st.title("vSolve: a Question Answering System")
        context = st.text_area(label="Context",value="",key="home_context",placeholder="Enter your context here...",height=200)
        question = st.text_input(label="Question",value="",key="home_question",placeholder="Enter your question here...")

        click = st.button(label='Get Answer',key="home_button")
        
        if click == True: 
            if len(context) == 0:
                st.warning('Context is missing. Context cannot be empty', icon="⚠️")
            if len(question) == 0:
                st.warning('Question is missing. Question cannot be empty', icon="⚠️")


            start_logits,end_logits,input_ids = model_functions.get_prediction(model,question,context,tokenizer)
            result = model_functions.postprocess(tokenizer,start_logits,end_logits,input_ids)  
            st.write(f"Answer: {result}")

    with ex1:
        st.title("Vidyalankar Institute of Technology")
        context = st.write(args.vit_info)
        question = st.text_input(label="Question",value="",placeholder="Enter your question here...",key="ex1_question")

        click = st.button(label='Get Answer',key="ex1_button")
        
        if click == True: 

            if len(question) == 0:
                st.warning('Question is missing. Question cannot be empty', icon="⚠️")


            start_logits,end_logits,input_ids = model_functions.get_prediction(model,question,args.vit_info,tokenizer)
            result = model_functions.postprocess(tokenizer,start_logits,end_logits,input_ids)  
            st.write(f"Answer: {result}")


    with ex2:
        st.title("Gateway of India")
        context = st.write(args.ex2_info)
        question = st.text_input(label="Question",value="",placeholder="Enter your question here...",key="ex2_question")

        click = st.button(label='Get Answer',key="ex2_button")
        
        if click == True: 

            if len(question) == 0:
                st.warning('Question is missing. Question cannot be empty', icon="⚠️")


            start_logits,end_logits,input_ids = model_functions.get_prediction(model,question,args.ex2_info,tokenizer)
            result = model_functions.postprocess(tokenizer,start_logits,end_logits,input_ids)  
            st.write(f"Answer: {result}")

        
