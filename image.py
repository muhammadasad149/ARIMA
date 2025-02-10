import streamlit as st
import base64
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_image_description(uploaded_file):
    encoded_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the following image in detail Analyze the given stock chart, which includes Bollinger Bands, RSI, and MACD indicators. Evaluate the trend direction, momentum, and potential reversal signals. Based on the technical indicators, determine whether it is a good time to enter a position or if the stock should be avoided. Conclude with a recommendation on whether to invest in this stock or not:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
                ],
            }
        ],
    )
    return response.choices[0].message.content

def main():
    st.title("Image Investment Insight Generator")
    st.write("Upload an image to get insights and investment advice.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                description = get_image_description(uploaded_file)
                
                st.subheader("Image Insights:")
                st.write(description)
                
                if "positive" in description.lower():
                    st.success("Based on the image analysis, this seems like a good investment opportunity!")
                else:
                    st.warning("Caution! The insights suggest that this might not be a great investment.")

if __name__ == "__main__":
    main()
