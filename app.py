# Set up Streamlit app in a Python file
from pyngrok import ngrok
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
import openai
from openai import OpenAI, AzureOpenAI
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
import pickle
import base64
from io import BytesIO
from transformers import pipeline
from textblob import TextBlob

# Load environment variables
load_dotenv()

# 1. ALWAYS THE FIRST STREAMLIT COMMAND:
st.set_page_config(page_title="Interview Topic Analysis", layout="wide")

# For deployment on Streamlit Community Cloud

# --- NLTK Data Management --- #
# Define a directory to store NLTK data within the app's environment
# This will be created in the application's root directory on Streamlit Community Cloud
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')

# Add the directory to NLTK's data path
# This ensures NLTK looks for data here first
nltk.data.path.append(nltk_data_dir)

# Create the directory if it doesn't exist
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download NLTK resources if they are not already present
# Use st.cache_resource to avoid re-downloading on every rerun if possible,
# though the try-except block handles this more fundamentally.
@st.cache_resource
def download_nltk_data():
    # Attempt to download 'punkt_tab' as explicitly requested by the ensuing error
    try:
        nltk.data.find('tokenizers/punkt_tab') # Check for punkt_tab specifically
    except LookupError:
        print("Downloading punkt_tab tokenizer...") # For debugging in logs
        nltk.download('punkt_tab', download_dir=nltk_data_dir) # Download punkt_tab

    # Keep the original 'punkt' download as well, just in case,
    # while recognising that 'punkt_tab' seems to be the direct missing one.
    # If the app starts working with just 'punkt_tab', we might remove this block.

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading punkt tokenizer...") # For debugging in logs
        nltk.download('punkt', download_dir=nltk_data_dir)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading stopwords corpus...")
        nltk.download('stopwords', download_dir=nltk_data_dir)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading wordnet corpus...")
        nltk.download('wordnet', download_dir=nltk_data_dir)

# Call the function to ensure data is present
download_nltk_data()

# --- End NLTK Data Management ---

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

st.title("Interview Topic Analysis Dashboard")
st.markdown("""
This app analyses interview data to extract topics using Latent Dirichlet Allocation (LDA).
Upload your dataset with questions, interview IDs, and answers to get started.
""")

if 'first_visit' not in st.session_state:
    st.session_state.first_visit = True

# Initialise Openai clients globally as none
client = None  # Regular OpenAI client
azure_client = None  # Azure OpenAI client

with st.sidebar:
    st.header("API Configuration")
    
    # API selection
    api_type = st.radio("Select API", ["OpenAI", "Azure OpenAI (Copilot)"])
    
    if api_type == "OpenAI":
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        if api_key:
            client = OpenAI(api_key=api_key)
            st.success("API key set!")
        else:
            st.warning("Please enter your OpenAI API key to use GPT features")

        # Choose a GPT Model
        gpt_model_choice = st.radio(
            "Choose GPT Model:",
            ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
            index = 0
        )
    else:  # Azure OpenAI
        azure_api_key = st.text_input("Enter Azure OpenAI API Key", type="password")
        azure_endpoint = st.text_input("Enter Azure Endpoint", 
                                     placeholder="https://your-resource.openai.azure.com")
        if azure_api_key and azure_endpoint:
            try:
                azure_client = AzureOpenAI(
                    api_key=azure_api_key,
                    api_version="2023-12-01-preview",
                    azure_endpoint=azure_endpoint
                )
                st.success("Azure OpenAI configuration set!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Introduce upload code
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
# Initialise Azure OpenAI client (for Copilot access)
def init_azure_openai_client(api_key=None, api_endpoint=None):
    if api_key and api_endpoint:
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2023-12-01-preview",  # Use the latest API version
            azure_endpoint=api_endpoint
        )
        return client
    elif "AZURE_OPENAI_API_KEY" in os.environ and "AZURE_OPENAI_ENDPOINT" in os.environ:
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2023-12-01-preview",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
        )
        return client
    return None
    
# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenise
        words = word_tokenize(text)
        # Remove stopwords and lemmatise
        processed_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
        return ' '.join(processed_words)
    return ""

# Function to compute coherence scores for different numbers of topics
def compute_coherence_values(dictionary, corpus, texts, start=2, limit=15, step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = models.LdaModel(corpus=corpus,
                               id2word=dictionary,
                               num_topics=num_topics,
                               random_state=42,
                               passes=10)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    
    return model_list, coherence_values

# Function to Label Topics using GPT
def label_topics_with_gpt(topics, model=None):
    global client
    
    if client is None:
        st.error("OpenAI API key not configured. Please enter it in the sidebar.")
        return None

    # Default model is gpt-4o unless overridden
    model_to_use = model or "gpt-4o"

    try:
        st.info(f"Using model: **{model_to_use}**")

        prompt = (
            f"Given the following LDA topic words, provide a short, meaningful label for each topic "
            f"(max 3-4 words):\n\n{topics}\n\nLabels:"
        )

        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": "You are an expert in topic modeling and text analysis. Your task is to provide concise, meaningful labels for topics based on the most representative words."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        if "gpt-4o" in model_to_use:
            st.warning("`gpt-4o` unavailable. Falling back to `gpt-3.5-turbo`.")
            return label_topics_with_gpt(topics, model="gpt-3.5-turbo")
        else:
            st.error(f"Error labelling topics with GPT: {e}")
            return None

def create_topic_viz(lda_model, corpus, dictionary):
    try:
        # Make sure to pass the correct parameters
        vis_data = pyLDAvis.gensim_models.prepare(
            lda_model, 
            corpus, 
            dictionary,
            sort_topics=False  # Try keeping original topic order
        )
        
        # Set parameters for HTML rendering
        html_string = pyLDAvis.prepared_data_to_html(
            vis_data,
            template_type='general',
            # Give specific ID
            visid='ldavis_html'
        )
        
        # Will use components.html to render it properly
        return html_string
    except Exception as e:
        st.error(f"Error creating topic visualization: {e}")
        # Consider logging the full traceback for debugging
        import traceback
        st.error(traceback.format_exc())
        return None

# Function to find optimal number of topics
def find_optimal_topics(coherence_values, topic_range):
    # Find the topic number with the highest coherence score
    max_index = coherence_values.index(max(coherence_values))
    return topic_range[max_index]

# Function to perform sentiment analysis
def analyze_topic_sentiment(lda_model, corpus, texts):
    """
    Analyse sentiment for each topic in the LDA model.
    
    Parameters:
    - lda_model: The trained LDA model
    - corpus: The document-term matrix
    - texts: List of original texts for sentiment analysis
    
    Returns:
    - DataFrame with topic distributions and sentiment scores
    """
    # Get topic distribution for each document
    topic_distributions = []
    for i, doc in enumerate(corpus):
        # Get topic distribution for document
        topic_dist = lda_model.get_document_topics(doc)
        # Convert to dict for easier processing
        topic_dict = {topic_id: prob for topic_id, prob in topic_dist}
        
        # Add the original text and sentiment
        sentiment = TextBlob(texts[i]).sentiment
        
        # Create a row with topic probabilities and sentiment
        row = {
            'document_id': i,
            'text': texts[i],
            'polarity': sentiment.polarity,
            'subjectivity': sentiment.subjectivity
        }
        
        # Add topic probabilities
        for topic_id in range(lda_model.num_topics):
            row[f'topic_{topic_id+1}'] = topic_dict.get(topic_id, 0.0)
            
        topic_distributions.append(row)
    
    # Convert to DataFrame
    df_topics = pd.DataFrame(topic_distributions)
    return df_topics

# Function to visualise sentiment by topic
def visualize_topic_sentiment(sentiment_df, lda_model):
    """
    Create visualisations for sentiment analysis by topic.
    
    Parameters:
    - sentiment_df: DataFrame with topic distributions and sentiment scores
    - lda_model: The trained LDA model
    
    Returns:
    - Dictionary of plotly figures
    """
    figures = {}
    
    # Get the dominant topic for each document
    topic_cols = [col for col in sentiment_df.columns if col.startswith('topic_')]
    sentiment_df['dominant_topic'] = sentiment_df[topic_cols].idxmax(axis=1)
    
    # Replace topic_N with just the number for better display
    sentiment_df['dominant_topic'] = sentiment_df['dominant_topic'].str.replace('topic_', 'Topic ')
    
    # Calculate average sentiment by topic
    topic_sentiment = sentiment_df.groupby('dominant_topic').agg({
        'polarity': 'mean',
        'subjectivity': 'mean',
        'document_id': 'count'
    }).reset_index()
    topic_sentiment = topic_sentiment.rename(columns={'document_id': 'document_count'})
    
    # Create polarity by topic bar chart
    polarity_fig = px.bar(
        topic_sentiment,
        x='dominant_topic',
        y='polarity',
        color='polarity',
        color_continuous_scale=px.colors.diverging.RdBu,
        title='Average Sentiment Polarity by Topic',
        labels={'dominant_topic': 'Topic', 'polarity': 'Sentiment Polarity'},
        text='document_count'
    )
    polarity_fig.update_traces(texttemplate='%{text} docs', textposition='outside')
    figures['polarity'] = polarity_fig
    
    # Create subjectivity by topic bar chart
    subjectivity_fig = px.bar(
        topic_sentiment,
        x='dominant_topic',
        y='subjectivity',
        color='subjectivity',
        color_continuous_scale='viridis',
        title='Average Subjectivity by Topic',
        labels={'dominant_topic': 'Topic', 'subjectivity': 'Subjectivity'},
        text='document_count'
    )
    subjectivity_fig.update_traces(texttemplate='%{text} docs', textposition='outside')
    figures['subjectivity'] = subjectivity_fig
    
    # Create scatter plot of polarity vs subjectivity by topic
    scatter_fig = px.scatter(
        sentiment_df,
        x='polarity',
        y='subjectivity',
        color='dominant_topic',
        title='Sentiment Distribution by Topic',
        labels={'polarity': 'Sentiment Polarity', 'subjectivity': 'Subjectivity'},
        hover_data=['text']
    )
    figures['scatter'] = scatter_fig
    
    return figures

def display_sentiment_analysis(results, unique_id = None):
    st.subheader("Sentiment Analysis by Topic")

    if 'sentiment_figures' in results:
        if 'polarity' in results['sentiment_figures']:
            st.plotly_chart(results['sentiment_figures']['polarity'])
        else:
            st.warning("Sentiment polarity figures not found in results.")

        if 'subjectivity' in results['sentiment_figures']:
            st.plotly_chart(results['sentiment_figures']['subjectivity'])
        else:
            st.warning("Sentiment subjectivity figures not found in results.")
    
        # Create tabs for detailed view
        sentiment_tab1, sentiment_tab2 = st.tabs(["Sentiment Distribution", "Raw Sentiment Data"])
    
        with sentiment_tab1:
            if 'scatter' in results['sentiment_figures']:
                st.plotly_chart(results['sentiment_figures']['scatter'])
            else:
                st.warning("Sentiment scatter plot not found in the results.")
    
        with sentiment_tab2:
            # Show the raw sentiment data
            st.write("Sentiment scores and topic distributions by document:")
            if 'sentiment_results' in results and not results['sentiment_results'].empty:
                # Format the dataframe for display
                display_cols = ['document_id', 'polarity', 'subjectivity', 'dominant_topic']
                topic_cols = [col for col in results['sentiment_results'].columns if col.startswith('topic_')]
                display_df = results['sentiment_results'][display_cols + topic_cols].copy()
        
                # Highlight the sentiment values
                st.dataframe(display_df.style.background_gradient(
                    subset=['polarity'], cmap='RdBu', vmin=-1, vmax=1
                ).background_gradient(
                    subset=['subjectivity'], cmap='viridis', vmin=0, vmax=1
                ))
        
                # Add download button for the sentiment data
                csv = display_df.to_csv(index=False)
                button_key = f"sentiment_download_{unique_id}" if unique_id else "sentiment_download"
                st.download_button(
                    label="Download Sentiment Data",
                    data=csv,
                    file_name="topic_sentiment_analysis.csv",
                    mime="text/csv",
                    # Add the unique key
                    key = button_key
                )
            else:
                st.info("Sentiment results dataframe is not available.")
    else:
        st.info("Sentiment analysis figures and results are not available.")

# Main analysis function
def analyze_topics(df, interview_id=None):
    if interview_id:
        # Filter data for specific interview
        filtered_df = df[df['InterviewID'] == interview_id]
    else:
        filtered_df = df
    # Test code
    if filtered_df.empty:
        return {'error': 'No data found for the selected interview.'}

    # Combine all answers for the selected interview
    all_answers = ' '.join(filtered_df['Answers'].astype(str).tolist())
    
    # Preprocess text
    processed_text = preprocess_text(all_answers)
    
    # Create a list of tokenized answers for coherence calculation
    tokenized_answers = [preprocess_text(answer).split() for answer in filtered_df['Answers'].astype(str).tolist() if isinstance(answer, str)]

    if len(tokenized_answers) < 2:
        return {'warning': 'Topic analysis requires at least two responses. Skipping topic analysis.'}
    
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(tokenized_answers)
    corpus = [dictionary.doc2bow(text) for text in tokenized_answers]
    
    # Compute coherence values for different numbers of topics
    topic_range = range(2, min(15, len(tokenized_answers) + 1), 1)
    model_list, coherence_values = compute_coherence_values(dictionary, corpus, tokenized_answers, 
                                                          start=topic_range[0], 
                                                          limit=topic_range[-1]+1, 
                                                          step=1)
    
    # Find optimal number of topics
    optimal_num_topics = find_optimal_topics(coherence_values, list(topic_range))
    
    # Create LDA model with optimal number of topics
    lda_model = models.LdaModel(corpus=corpus,
                               id2word=dictionary,
                               num_topics=optimal_num_topics, 
                               random_state=42,
                               passes=10)
    
    # Get topics and their words
    topics = lda_model.print_topics(num_words=8)
    
    # Format topics for GPT labelling
    topic_str = "\n".join([f"Topic {i+1}: {topic[1]}" for i, topic in enumerate(topics)])

    # Get original texts for sentiment analysis
    original_texts = df['Answers'].astype(str).tolist() if interview_id is None else df[df['InterviewID'] == interview_id]['Answers'].astype(str).tolist()
    
    # Perform sentiment analysis
    sentiment_results = analyze_topic_sentiment(lda_model, corpus, original_texts)
    
    # Generate sentiment visualisations
    sentiment_figures = visualize_topic_sentiment(sentiment_results, lda_model)
    
    return {
        'lda_model': lda_model,
        'corpus': corpus,
        'dictionary': dictionary,
        'topics': topics,
        'topic_str': topic_str,
        'coherence_values': coherence_values,
        'topic_range': list(topic_range),
        'optimal_num_topics': optimal_num_topics,
        'tokenized_answers': tokenized_answers,
        'sentiment_results': sentiment_results,
        'sentiment_figures': sentiment_figures
    }

def explain_topic(model, topic_id, top_n=10):
    """Helper function to explain a specific topic in detail"""
    
    st.subheader(f"Detailed Analysis of Topic {topic_id}")
    
    # Get the top terms for this topic
    topic_terms = model.show_topic(topic_id, topn=top_n)
    
    # Display terms and their weights
    term_df = pd.DataFrame(topic_terms, columns=["Term", "Weight"])
    st.dataframe(term_df)
    
    # Show topic distribution across documents
    st.write("##### Topic Distribution in Documents")
    # Add code to show in which documents this topic is prominent
    
    # Explain coherence
    coherence = model.top_topics(corpus)[topic_id][1]
    st.write(f"Topic coherence score: {coherence:.4f}")
    
    # Interpretation tips
    st.info("""
    **How to interpret:** Higher coherence scores suggest more semantically coherent topics.
    Look for distinct patterns in term weights - sharp drop-offs suggest more focused topics.
    """)

def get_markdown_download_link(markdown_content, filename="pyLDAvis_documentation.md"):
    """Generate a link to download the markdown content as a file"""
    
    # Encode the markdown content
    b64 = base64.b64encode(markdown_content.encode()).decode()
    
    # Create the download link
    href = f'<a href="data:file/markdown;base64,{b64}" download="{filename}">Download Markdown Documentation</a>'
    
    return href

# Define your sentiment analysis documentation content
sentiment_docs = """
# Understanding Sentiment Analysis

## Overview
Sentiment analysis identifies and extracts subjective information from text to determine the writer's attitude toward a topic—whether it's positive, negative, or neutral. This documentation explains how our sentiment analysis works, how to interpret results, and common patterns to look for.

## Key Components

### Sentiment Scores
- **Positive score**: Indicates the strength of positive sentiment (typically 0-1)
- **Negative score**: Indicates the strength of negative sentiment (typically 0-1)
- **Neutral score**: Indicates the strength of neutral sentiment (typically 0-1)
- **Compound score**: A normalized score that combines all three dimensions (-1 to +1)

### Visualization Elements
- **Bar charts**: Show the distribution of sentiments across your dataset
- **Line graphs**: Display sentiment trends over time (if time data is available)
- **Heatmaps**: Identify correlations between sentiment and other variables
- **Word clouds**: Highlight terms associated with different sentiment categories

## Interpreting the Results

### Compound Score Interpretation
- **-1.0 to -0.05**: Negative sentiment
- **-0.05 to +0.05**: Neutral sentiment
- **+0.05 to +1.0**: Positive sentiment

The further the score is from zero, the stronger the sentiment.

### Common Patterns to Look For

1. **Sentiment distribution**
   - Is your data predominantly positive, negative, or neutral?
   - Are there unexpected sentiment clusters?

2. **Sentiment extremes**
   - Which texts have the most extreme positive or negative scores?
   - What themes appear in these extreme cases?

3. **Sentiment vs. topics**
   - Do certain topics correlate with specific sentiment patterns?
   - Are some topics more polarizing than others?

4. **Context dependence**
   - Sentiment can vary by domain (e.g., "unpredictable" may be negative for product reviews but positive for movie reviews)
   - Industry-specific terms may carry sentiment not recognized by general models

## Limitations and Considerations

### Model Limitations
- **Sarcasm and irony**: Most sentiment models struggle with these nuances
- **Domain specificity**: General models may miss domain-specific sentiment
- **Contextual understanding**: Models may not fully grasp contextual shifts in meaning

### Interpreting Edge Cases

1. **Mixed sentiment texts**
   - Texts with both strong positive and negative elements may average to neutral
   - Consider examining positive and negative scores separately for these cases

2. **Neutral classifications**
   - May indicate truly neutral content OR balanced positive/negative content
   - Check individual positive/negative scores to distinguish these cases

3. **Short texts**
   - Brief texts may have less reliable sentiment scores
   - Consider aggregating short texts for more reliable analysis
"""

# Main app logic
if uploaded_file is not None:
    # Load and display data
    df = pd.read_csv(uploaded_file)
    
    # Display raw data
    st.subheader("Raw Data")
    st.write(df.head())
    
    # Get unique interview IDs
    interview_ids = df['InterviewID'].unique()
    
    # Display analysis options
    st.subheader("Analysis Options")
    
    analysis_option = st.radio(
        "Choose analysis type:",
        ["All interviews combined", "Individual interview", 
         "Group by InterviewID"]
    )
    
    if analysis_option == "All interviews combined":
        if st.button("Run Combined Analysis"):
            with st.spinner("Running topic analysis on all interviews..."):
                results = analyze_topics(df)
                # Store results in session state for access across the app
                st.session_state.results = results
                
                # Display optimal number of topics
                st.subheader("Topic Modelling Results")
                st.write(f"**Optimal number of topics:** {results['optimal_num_topics']}")
                
                # Plot coherence scores
                fig = px.line(
                    x=results['topic_range'],
                    y=results['coherence_values'],
                    labels={'x': 'Number of Topics', 'y': 'Coherence Score'},
                    title='Topic Coherence Scores'
                )
                fig.add_vline(x=results['optimal_num_topics'], line_dash="dash", line_color="red")
                st.plotly_chart(fig)
                
                # Display topics
                st.subheader("Generated Topics")
                for i, topic in enumerate(results['topics']):
                    st.write(f"**Topic {i+1}:** {topic[1]}")
                
                # Get topic labels using GPT
                if api_key:
                    with st.spinner("Labelling topics with GPT..."):
                        gpt_labels = label_topics_with_gpt(results['topic_str'], model = gpt_model_choice)
                        if gpt_labels:
                            st.subheader("GPT-Generated Topic Labels")
                            st.write(gpt_labels)
                            
                            # Let user edit or refine labels
                            edited_labels = st.text_area("Edit or refine these labels:", gpt_labels)
                else:
                    st.warning("Please enter an OpenAI API key in the sidebar to enable GPT topic labeling.")
                
                # Create and display interactive topic visualisation
                if st.session_state.first_visit:
                    st.info(" **First time using the Topic Visualizer?** Follow the brief tour below to understand what you're seeing.")
    
                    st.markdown("""
                    ### Brief Tour of the Topic Visualization
                    1. **Left panel**: Each bubble is a topic. Click on any bubble to explore that topic.
                    2. **Right panel**: See which terms define each topic.
                    3. **Try this**: Click different bubbles and watch how the red bars change.
                    4. **Experiment**: Adjust the λ slider to see different perspectives on terms.
                    """)
    
                    if st.button("Got it! Don't show again"):
                        st.session_state.first_visit = False
                        st.experimental_rerun()
                
                st.subheader("Interactive Topic Visualisation")
                html_viz = create_topic_viz(results['lda_model'], results['corpus'], results['dictionary'])
                if html_viz:
                    # Add expandable help section before the visualisation
                    with st.expander(" How to interpret this visualisation"):
                        st.markdown("""
                        ### Quick Guide
                        - **Topic bubbles**: Size shows topic prevalence, distance shows semantic similarity
                        - **Red bars**: Show term relevance to the selected topic (missing bars indicate shared terms)
                        - **Blue bars**: Show overall term frequency (missing bars indicate rare terms)
                        - **Lambda (λ) slider**: Adjust to balance term probability vs. distinctiveness
    
                        [Click here to learn more](https://pyldavis.readthedocs.io/en/latest/) about topic modelling visualisations.
                        """)
    
                    # Display the visualisation
                    st.components.v1.html(html_viz, height=800)
        
                with st.expander(" How to interpret sentiment analysis"):
                    st.markdown("""
                    ### Quick Guide
                    - Compound score: Overall sentiment from -1 (negative) to +1 (positive)
                    - Positive/Negative/Neutral scores: Individual sentiment dimensions (0-1)
                    - Thresholds: Negative < -0.05, Neutral -0.05 to +0.05, Positive > +0.05
                    
                    Download the complete documentation for detailed explanations.
                    """)
                # Display sentiments
                display_sentiment_analysis(results, unique_id = "combined")

                st.markdown("### Documentation Resources: Sentiment Analysis")
                #st.markdown(get_markdown_download_link(sentiment_docs), unsafe_allow_html=True)
                
                # Documentation download buttons
                col1, col2 = st.columns(2)
    
                with col1:
                    st.download_button(
                        label=" Download Full Sentiment Analysis Documentation",
                        data=sentiment_docs,
                        file_name="Sentiment_Analysis_Guide.md",
                        mime="text/markdown",
                        help="Download the complete sentiment analysis documentation in markdown format"
                        )
    
                with col2:
                    st.download_button(
                        label=" Download as Plain Text",
                        data=sentiment_docs,
                        file_name="Sentiment_Analysis_Guide.txt",
                        mime="text/plain",
                        help="Download documentation as plain text"
                        )
    
    elif analysis_option == "Individual interview":
        selected_interview = st.selectbox("Select Interview ID:", interview_ids)
        
        if st.button("Run Interview Analysis"):
            with st.spinner(f"Running topic analysis for Interview {selected_interview}..."):
                results = analyze_topics(df, selected_interview)

                if 'error' in results:
                    st.error(results['error'])
                elif 'warning' in results:
                    st.warning(results['warning'])
                else:
                    # Display optimal number of topics
                    st.subheader(f"Topic Modelling Results for Interview {selected_interview}")
                    st.write(f"**Optimal number of topics:** {results['optimal_num_topics']}")
                
                    # Plot coherence scores
                    fig = px.line(
                        x=results['topic_range'],
                        y=results['coherence_values'],
                        labels={'x': 'Number of Topics', 'y': 'Coherence Score'},
                        title=f'Topic Coherence Scores for Interview {selected_interview}'
                    )
                    fig.add_vline(x=results['optimal_num_topics'], line_dash="dash", line_color="red")
                    st.plotly_chart(fig)
                
                    # Display topics
                    st.subheader("Generated Topics")
                    for i, topic in enumerate(results['topics']):
                        st.write(f"**Topic {i+1}:** {topic[1]}")
                
                    # Get topic labels using GPT
                    if api_key:
                        with st.spinner("Labelling topics with GPT..."):
                            gpt_labels = label_topics_with_gpt(results['topic_str'], model = gpt_model_choice)
                            if gpt_labels:
                                st.subheader("GPT-Generated Topic Labels")
                                st.write(gpt_labels)
                            
                                # Let user edit or refine labels
                                edited_labels = st.text_area("Edit or refine these labels:", gpt_labels)
                    else:
                        st.warning("Please enter an OpenAI API key in the sidebar to enable GPT topic labeling.")
                
                    # Create and display interactive topic visualisation
                    st.subheader("Interactive Topic Visualisation")
                    html_viz = create_topic_viz(results['lda_model'], results['corpus'], results['dictionary'])
                    if html_viz:
                        # Add expandable help section before the visualisation
                        with st.expander("How to interpret this visualisation"):
                            st.markdown("""
                            ### Quick Guide
                            - **Topic bubbles**: Size shows topic prevalence, distance shows semantic similarity
                            - **Red bars**: Show term relevance to the selected topic (missing bars indicate shared terms)
                            - **Blue bars**: Show overall term frequency (missing bars indicate rare terms)
                            - **Lambda (λ) slider**: Adjust to balance term probability vs. distinctiveness
    
                            [Click here to learn more](https://pyldavis.readthedocs.io/en/latest/) about topic modelling visualisations.
                            """)
                            
                        st.components.v1.html(html_viz, height = 800)
    
                    with st.expander("How to interpret sentiment analysis"):
                        st.markdown("""
                        ### Quick Guide
                        - Compound score: Overall sentiment from -1 (negative) to +1 (positive)
                        - Positive/Negative/Neutral scores: Individual sentiment dimensions (0-1)
                        - Thresholds: Negative < -0.05, Neutral -0.05 to +0.05, Positive > +0.05
                        
                        Download the complete documentation for detailed explanations.
                        """)
                        
                    # Display sentiments
                    display_sentiment_analysis(results, unique_id=f"interview_{selected_interview}")

                    st.markdown("Download Full Documentation: Sentiment Analysis")
                    # Documentation download buttons
                    col1, col2 = st.columns(2)

                    with col1:
                        st.download_button(
                        label=" Download Full Documentation",
                        data=sentiment_docs,
                        file_name="Sentiment_Analysis_Guide.md",
                        mime="text/markdown",
                        help="Download the complete sentiment analysis documentation in markdown format"
                        )

                    with col2:
                        st.download_button(
                        label=" Download as Plain Text",
                        data=sentiment_docs,
                        file_name="Sentiment_Analysis_Guide.txt",
                        mime="text/plain",
                        help="Download documentation as plain text"
                        )
        
    elif analysis_option == "Group by InterviewID":
        if st.button("Run Group Analysis"):
            # Create tabs for comparison
            tab1, tab2 = st.tabs(["Optimal Topics by Interview", "Coherence Comparison"])
            
            with st.spinner("Analysing topics across all interviews..."):
                all_results = {}
                optimal_topics = {}
                coherence_data = []
                
                # Process each interview
                for interview_id in interview_ids:
                    results = analyze_topics(df, interview_id)

                    all_results[interview_id] = results

                    if 'optimal_num_topics' in results:
                        optimal_topics[interview_id] = results['optimal_num_topics']
                    
                        # Collect coherence data for comparison
                        for i, num_topics in enumerate(results['topic_range']):
                            coherence_data.append({
                                'InterviewID': interview_id,
                                'Number of Topics': num_topics,
                                'Coherence Score': results['coherence_values'][i]
                            })
                    elif 'warning' in results:
                        st.warning(f"Interview {interview_id}:{results['warning']}")
                    elif 'error' in results:
                        st.error(f"Interview {interview_id}: {results['error']}")
                
                # Display optimal topics by interview
                with tab1:
                    st.subheader("Optimal Number of Topics by Interview")
                    if optimal_topics:
                        # Create bar chart
                        fig = px.bar(
                            x=list(optimal_topics.keys()),
                            y=list(optimal_topics.values()),
                            labels={'x': 'Interview ID', 'y': 'Optimal Number of Topics'},
                            title='Optimal Number of Topics by Interview'
                        )
                        st.plotly_chart(fig)
                    
                        # Display as table
                        opt_topics_df = pd.DataFrame({
                            'InterviewID': list(optimal_topics.keys()),
                            'Optimal Number of Topics': list(optimal_topics.values())
                        })
                        st.write(opt_topics_df)
                    else:
                        st.info("No interviews had enough responses for topic analysis.")
                
                # Display coherence comparison
                with tab2:
                    st.subheader("Coherence Score Comparison")

                    if coherence_data: 
                        coherence_df = pd.DataFrame(coherence_data)
                    
                        # Create line chart for coherence comparison
                        fig = px.line(
                            coherence_df,
                            x='Number of Topics',
                            y='Coherence Score',
                            color='InterviewID',
                            title='Coherence Scores Across Interviews'
                        )
                        st.plotly_chart(fig)
                    else:
                        st.info("Coherence data is not available due to insufficient responses in some interviews.")
                
                # Create detailed analysis for each interview
                st.subheader("Detailed Analysis by Interview")
                
                for interview_id, results in all_results.items():
                    #with st.expander(f"Interview {interview_id} - {results['optimal_num_topics']} Topics"):
                    with st.expander(f"Interview {interview_id} - {'{} Topics'.format(results.get('optimal_num_topics', 'N/A'))}"):    
                        if 'topics' in results:
                            # Display topics      
                            st.write("**Topics:**")
                            for i, topic in enumerate(results['topics']):
                                st.write(f"Topic {i+1}: {topic[1]}")
                        
                            # Get topic labels using GPT if API key is provided
                            if api_key and 'topic_str' in results:
                                with st.spinner(f"Labelling topics for Interview {interview_id}..."):
                                    gpt_labels = label_topics_with_gpt(results['topic_str'], model = gpt_model_choice)
                                    if gpt_labels:
                                        st.write("**GPT-Generated Labels:**")
                                        st.write(gpt_labels)
                            elif api_key:
                                st.warning("Topic string not available for GPT labelling.")
                            else:
                                st.warning("Please enter an OpenAI API key in the sidebar to enable GPT topic labelling.")
                        
                        if 'lda_model' in results and 'corpus' in results and 'dictionary' in results:
                            # Display interactive visualisation
                            html_viz = create_topic_viz(results['lda_model'], results['corpus'], results['dictionary'])
                            if html_viz:
                                # Add expandable help section before the visualization
                                st.subheader(" How to interpret this visualisation")
                                st.markdown("""
                                    ### Quick Guide
                                    - **Topic bubbles**: Size shows topic prevalence, distance shows semantic similarity
                                    - **Red bars**: Show term relevance to the selected topic (missing bars indicate shared terms)
                                    - **Blue bars**: Show overall term frequency (missing bars indicate rare terms)
                                    - **Lambda (λ) slider**: Adjust to balance term probability vs. distinctiveness
                                
                                    [Click here to learn more](https://pyldavis.readthedocs.io/en/latest/) about topic modeling visualizations.
                                    """)
    
                            # Display the visualisation
                            st.components.v1.html(html_viz, height=600)

                        elif 'warning' in results:
                            st.warning(f"Interview {interview_id}: {results['warning']}")
                        elif 'error' in results:
                            st.error(f"Interview {interview_id}: {results['error']}")
                        
                        st.subheader(" How to interpret sentiment analysis")
                        st.markdown("""
                        ### Quick Guide
                        - Compound score: Overall sentiment from -1 (negative) to +1 (positive)
                        - Positive/Negative/Neutral scores: Individual sentiment dimensions (0-1)
                        - Thresholds: Negative < -0.05, Neutral -0.05 to +0.05, Positive > +0.05
                            
                        Download the complete documentation for detailed explanations.
                        """)
                            
                        # Displaying Sentiment Analysis Results
                        display_sentiment_analysis(results, unique_id=f"group_{interview_id}")

#                        st.markdown("Download Documentation: Sentiment Analysis")
#
#                        # Documentation download buttons
#                        col1, col2 = st.columns(2)
#
#                        with col1:
#                            st.download_button(
#                            label=" Download Full Documentation",
#                            data=sentiment_docs,
#                            file_name="Sentiment_Analysis_Guide.md",
#                            mime="text/markdown",
#                            help="Download the complete sentiment analysis documentation in markdown format"
#                            )
#
#                        with col2:
#                            st.download_button(
#                            label=" Download as Plain Text",
#                            data=sentiment_docs,
#                            file_name="Sentiment_Analysis_Guide.txt",
#                            mime="text/plain",
#                            help="Download documentation as plain text"
#                            )

# Add a custom topic labelling section
st.subheader("Custom Topic Labelling")

with st.expander("Label topics manually or with AI assistants"):
    topic_input = st.text_area("Enter topic words to label (each topic on a new line):", 
                             placeholder="Topic 1: 0.010*\"data\" + 0.008*\"analysis\" + 0.006*\"research\" + 0.005*\"information\"")
    
    assistant_choice = st.radio("Choose AI assistant for labelling:", ["GPT (OpenAI)", "Copilot (if configured)"])

    # Model selection for GPT
    gpt_model = None
    if assistant_choice == "GPT (OpenAI)":
        gpt_model = st.selectbox(
            "Choose GPT model:",
            options=["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
            index=0
        )
    
    if st.button("Generate Labels"):
        if topic_input:
            if assistant_choice == "GPT (OpenAI)":
                if api_key:
                    with st.spinner("Generating labels with GPT..."):
                        gpt_labels = label_topics_with_gpt(topic_input, model = gpt_model)
                        if gpt_labels:
                            st.write("**Generated Labels:**")
                            st.write(gpt_labels)
                            
                            # Add option to save labels
                            st.download_button(
                                label="Download Labels",
                                data=gpt_labels,
                                file_name="topic_labels.txt",
                                mime="text/plain"
                            )
                else:
                    st.warning("Please enter an OpenAI API key in the sidebar to use GPT labeling.")
            elif assistant_choice == "Copilot (if configured)":
                st.info("Copilot integration would require additional implementation with your specific Copilot setup.")
                st.write("This would typically involve setting up an API connection to your Copilot endpoint.")
        else:
            st.warning("Please enter topic words to label.")

######################################################################
with st.sidebar:
    st.header("Save/Load Analysis")
    
    # Save functionality
    if st.button("Save Current Analysis"):
        # Debug: Show what variables are available
        #st.write("Debug - Session state keys:", list(st.session_state.keys()))
        #st.write("Debug - Global variables:", [k for k in globals().keys() if not k.startswith('_')])
        
        # Check for results in various possible places
        results_to_save = None
        
        # Try different common variable names
        possible_names = ['results', 'analysis_results', 'topic_results', 'model_results']
        
        for name in possible_names:
            if name in st.session_state:
                results_to_save = st.session_state[name]
                st.write(f"Found results in session_state.{name}")
                break
            elif name in globals():
                results_to_save = globals()[name]
                st.write(f"Found results in globals().{name}")
                break
        
        if results_to_save is not None:
            try:
                buffer = BytesIO()
                pickle.dump(results_to_save, buffer)
                buffer.seek(0)
                st.download_button(
                    label="Download Analysis",
                    data=buffer,
                    file_name="topic_analysis.pkl",
                    mime="application/octet-stream"
                )
            except Exception as e:
                st.error(f"Error saving analysis: {e}")
        else:
            st.error("No analysis results found to save.")
    
    # Load functionality
    saved_model = st.file_uploader("Load Saved Analysis", type=['pkl'])
    if saved_model:
        try:
            loaded_results = pickle.load(saved_model)
            # Store in both session state and global variable for compatibility
            st.session_state.results = loaded_results
            # Use globals() to set the variable in the global scope to maintain consistency
            globals()['results'] = loaded_results
            st.success("Analysis loaded successfully!")

            st.subheader("Loaded Analysis Contents:")
            if isinstance(loaded_results, dict):
                st.write("Loaded data is a dictionary.")
                for key, value in loaded_results.items():
                    st.write(f"**Key:** `{key}`")
                    st.write(f"**Type:** `{type(value)}`")
                    # Displaying different types
                    if isinstance(value, (str, int, float, bool)):
                        st.write(f"**Value:** `{value}`")
                    elif isinstance(value, (list, tuple, set)):
                        st.write(f"**Length:** `{len(value)}`")
                        st.json(list(value[:5])) # Show first 5 items if it's a list/tuple/set
                        if len(value) > 5:
                            st.write("...(showing first 5 items)")
                    elif hasattr(value, 'head') and hasattr(value, 'shape'): # Likely a Pandas DataFrame
                        st.write(f"**Shape:** `{value.shape}`")
                        st.dataframe(value.head())
                        st.write("...(showing first 5 rows)")
                    elif hasattr(value, '__dict__'): # Custom object
                        st.write(f"**Attributes:** `{value.__dict__.keys()}`")
                        # You might need to add specific display logic for your custom objects
                        # st.write(value) # This might print a messy representation
                    else:
                        st.write(f"**Value:** `{str(value)[:200]}`...") # Generic display for other types
            elif isinstance(loaded_results, (list, tuple, set)):
                st.write(f"Loaded data is a {type(loaded_results).__name__} with {len(loaded_results)} items.")
                st.json(list(loaded_results[:5])) # Show first 5 items
                if len(loaded_results) > 5:
                    st.write("...(showing first 5 items)")
            elif hasattr(loaded_results, 'head') and hasattr(loaded_results, 'shape'): # Likely a Pandas DataFrame
                st.write("Loaded data is likely a Pandas DataFrame.")
                st.write(f"**Shape:** `{loaded_results.shape}`")
                st.dataframe(loaded_results.head())
                st.write("...(showing first 5 rows)")
            else:
                st.write(f"Loaded data type: `{type(loaded_results)}`")
                st.write("Full content (may be truncated for large objects):")
                st.write(str(loaded_results)[:1000]) # Display a portion of the content
        except Exception as e:
            st.error(f"Error loading analysis: {e}")

#######################################################################
# Assuming you have loaded the analysis in the sidebar
# Now, in the main part of your app:

if 'results' in st.session_state and st.session_state.results is not None:
    st.title("Your Analysis Dashboard")

    results = st.session_state.results # Get the loaded results

    if 'topic_model' in results:
        st.subheader("Topic Model Details:")
        # Display some info about your topic model
        # e.g., st.write(f"Number of topics: {results['topic_model'].n_components}")
        st.write("Topic model object is present.")
        # You would interact with your topic model object here
        # Example: st.write(results['topic_model'].get_params())

    if 'documents_df' in results:
        st.subheader("Processed Documents:")
        st.dataframe(results['documents_df'].head())
        st.write(f"Total documents: {len(results['documents_df'])}")

    # Add more display logic based on what's in your 'results' object
else:
    st.info("Please load an analysis from the sidebar to view results.")

#######################################################################
# Markdown documentation content
pyldavis_docs = """
# Understanding pyLDAvis Visualisations

## Overview
The pyLDAvis visualisation displays topic modelling results in an interactive format. It helps you explore the topics discovered by the LDA algorithm and the terms most associated with each topic.

## Key Components

### Left Panel: Topic Bubbles
- Each bubble represents a topic
- Size of bubbles indicates the prevalence of topics in the corpus
- Distance between bubbles approximates the semantic relationship between topics
- Similar topics appear closer together

### Right Panel: Term Bars
- **Blue bars**: Overall term frequency in the corpus
- **Red bars**: Term relevance to the selected topic

## Interpreting the Visualisation

### Common Observations

1. **Missing red bars when hovering over a topic**
   - This is normal for topics that don't have strongly distinctive terms
   - It may indicate topics composed mostly of terms that appear across multiple topics
   - Not necessarily a problem, but could suggest this topic is less coherent

2. **Missing blue bars for certain terms**
   - Indicates terms with very low frequency in the overall corpus
   - These may be rare but meaningful terms for specific topics

3. **Overlapping topics**
   - Topics positioned close together likely share similar themes or vocabulary
   - This is expected in natural language where themes often blend together

### Adjusting the Relevance Metric (λ)

The slider labelled "λ" controls how terms are ranked:
- λ=1.0: Terms ranked purely by their probability within topics
- λ=0.0: Terms ranked by their distinctiveness to this topic
- Intermediate values balance both considerations

Adjust this slider to explore different aspects of your topics and discover terms that are most relevant or distinctive.

## Tips for Optimal Use

- Explore different topics by clicking on different bubbles
- Adjust the λ slider to find different perspectives on each topic
- Consider the visualisation alongside other evaluation metrics for a complete understanding
- Remember that topic coherence is subjective and context-dependent

## Technical Implementation

This visualisation is generated using pyLDAvis with the gensim-models adapter.
"""

# Display the Documentation Resources
with st.sidebar:
    st.markdown("### Documentation Resources")
    #st.markdown(get_markdown_download_link(pyldavis_docs), unsafe_allow_html=True)
    
    # For plain text option
    st.download_button(
        label="Download as Plain Text",
        data=pyldavis_docs,
        file_name="pyLDAvis_Guide.txt",
        mime="text/plain",
        help = "Download documentation as plain text"
    )

    # Simple direct download using Streamlit's built-in functionality
    st.download_button(
        label = "Download Documentation",
        data = pyldavis_docs,
        file_name = "pyLDAvis_Guide.md",
        mime = "text/markdown",
        help = "Download complete documentation in markdown format"
    )

# Link to a web version if you have one
st.markdown(
    "[ View Online Documentation](https://pyldavis.readthedocs.io/en/latest/)")

# Assuming you have loaded the analysis in the sidebar
# Now, in the main part of your app, display the results

if 'results' in st.session_state and st.session_state.results is not None:
    st.title("Your Analysis Dashboard")

    results = st.session_state.results # Get the loaded results

    if 'topic_model' in results:
        st.subheader("Topic Model Details:")
        # Display some info about your topic model
        # e.g., st.write(f"Number of topics: {results['topic_model'].n_components}")
        st.write("Topic model object is present.")
        # You would interact with your topic model object here
        # Example: st.write(results['topic_model'].get_params())

    if 'documents_df' in results:
        st.subheader("Processed Documents:")
        st.dataframe(results['documents_df'].head())
        st.write(f"Total documents: {len(results['documents_df'])}")

    # Add more display logic based on what's in your 'results' object
else:
    st.info("Please load an analysis from the sidebar to view results.")

# Add footer with information
st.markdown("---")
st.markdown("""
**About this app:**
This tool uses Latent Dirichlet Allocation (LDA) to extract topics from interview data. 
It automatically determines the optimal number of topics based on coherence scores and 
can leverage GPT or other AI assistants to generate human-readable labels for the topics.
""")
