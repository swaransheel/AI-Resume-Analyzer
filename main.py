import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from dotenv import load_dotenv
import tempfile
from typing import List
import logging
import pygame
import time
import re
from gtts import gTTS
import plotly.graph_objects as go
from google.cloud import texttospeech



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class AudioPlayer:
    def __init__(self):
        if not self.is_cloud_environment():
            pygame.mixer.init()
        else:
            logger.info("Audio playback disabled in Streamlit Cloud.")

    def is_cloud_environment(self) -> bool:
        """Check if the app is running on Streamlit Cloud."""
        return os.getenv("STREAMLIT_CLOUD") is not None

    def generate_speech(self, text: str, lang: str = 'en') -> str:
        """Generate speech from text and save it as a temporary MP3 file."""
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts = gTTS(text=text, lang=lang, tld='us')  # You can change the TLD if needed
            tts.save(temp_file.name)
            logger.info(f"Generated speech audio: {temp_file.name}")
            return temp_file.name
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            raise ValueError(f"Failed to generate speech audio: {str(e)}")

    def play_audio(self, file_path: str):
        """Play the generated audio file (local only)."""
        if self.is_cloud_environment():
            logger.info("Audio playback is not supported in the cloud.")
            return

        try:
            pygame.mixer.init()  # Ensure mixer is initialized
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error playing audio: {str(e)}")
            raise ValueError(f"Failed to play audio: {str(e)}")
        finally:
            pygame.mixer.music.unload()  # Unload music to free resources

    def cleanup(self, file_path: str):
        """Clean up the temporary audio file."""
        try:
            os.unlink(file_path)
            logger.info(f"Cleaned up audio file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up audio file: {str(e)}")
            raise ValueError(f"Error cleaning up audio file: {str(e)}")
class InterviewIntroGenerator:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.configure_genai()
        self.setup_streamlit_style()
        self.audio_player = AudioPlayer()
        
    def configure_genai(self):
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        
    @staticmethod
    def setup_streamlit_style():
        st.set_page_config(page_title="Interview Introduction Generator", page_icon="👔", layout="wide")
        
        st.markdown("""
            <style>
            .header-container {
                text-align: center;
                padding: 2rem 0;
                margin-bottom: 2rem;
            }
            .header-text {
                font-size: 2.5rem;
                font-weight: bold;
                color: brown;
            }
            .subheader-text {
                font-size: 1.2rem;
                color: brown;
                margin-top: 0.5rem;
            }
            .introduction-container {
                padding: 20px;
                border-radius: 10px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                margin: 10px 0;
                color: #000000;
                font-size: 1rem;

            }
            .introduction-container * {
            font-size: inherit;
            }
            .stButton button {
                width: 100%;
                background-color: #98d99a;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            </style>
            """, unsafe_allow_html=True)


    def calculate_ats_score(self, text: str) -> dict:
        """
        Calculate ATS compatibility scores using generative AI analysis based on universal resume standards.
        
        Args:
            text (str): The resume text content
            
        Returns:
            dict: Dictionary containing scores for different aspects of the resume
        """
        prompt_template = """
Scoring Instructions for Resume ATS Compatibility:

Evaluate the resume using a strict 0-100 point scale for each category. Provide ONLY the numerical score for each category, without additional commentary. Use precise, quantitative deduction criteria. The scoring system is designed so that **only excellent resumes** should score **above 70%**, with most resumes falling below that range unless they meet a very high standard of ATS optimization.

1. Keyword Matching [0-10]:
- Alignment with job-specific keywords and relevant industry terminology.
- High keyword density (without overstuffing) and strategic placement.
- Relevance and correct usage of keywords within the context of the resume.
Deduction Criteria:
- No keyword alignment or minimal relevance: -9 points
- Overuse of keywords or generic, non-specific keywords: -8 points
- Lack of industry-specific terminology: -7 points
- Keywords placed in irrelevant sections: -6 points
- Missed opportunities to align with expected industry terms: -5 points

2. Formatting and Readability [0-10]:
- ATS-friendly, standard fonts (simple and professional).
- Clear and consistent document structure.
- No use of tables, columns, or embedded graphics.
- Consistent formatting of headings, subheadings, and text sections.
Deduction Criteria:
- Non-standard fonts or complex designs (e.g., decorative fonts): -8 points
- Tables or columns present: -9 points
- Embedded images or graphics: -8 points
- Inconsistent or non-professional formatting (e.g., mismatched headings): -7 points
- Overcrowded content or excessive spacing: -6 points

3. File Format Compatibility [0-10]:
- Standard file formats (.docx, .pdf) that are ATS-friendly.
- The document should be text-based, not image-based (no scanned docs).
- The file should have no embedded, complex elements.
Deduction Criteria:
- Non-ATS-friendly formats (e.g., .odt, .rtf): -9 points
- Image-based documents (e.g., scanned PDFs): -9 points
- Complex embedded elements (e.g., macros, hidden metadata): -8 points
- Unreadable or corrupted file formats: -10 points

4. Contact Information Clarity [0-10]:
- Professional email address and complete contact details (location, phone, etc.).
- Clear, easily accessible contact section at the top.
Deduction Criteria:
- Missing contact information: -8 points
- Unprofessional email address (e.g., using a nickname): -7 points
- Incomplete location or contact info: -6 points
- Poorly formatted or hidden contact details: -5 points

5. Work Experience Structure [0-10]:
- Chronological or reverse-chronological order with clear job titles and achievements.
- Action-oriented language with measurable outcomes.
- Consistent and precise date formats.
Deduction Criteria:
- Unclear job progression or titles: -8 points
- No measurable achievements (i.e., no numbers or specific results): -9 points
- Inconsistent or missing date formats: -7 points
- Generic descriptions with no impact or value: -7 points
- Job roles or achievements not clearly separated by bullet points: -6 points

6. Education Details Precision [0-10]:
- Full degree information, institution names, graduation dates, and certifications listed.
Deduction Criteria:
- Missing degree details or ambiguous information: -8 points
- Incomplete institution names or locations: -7 points
- Missing graduation dates: -8 points
- Lack of certifications or relevant education details: -9 points

7. Skills Section Comprehensiveness [0-10]:
- Listing of relevant skills, both hard and soft.
- Technical skills clearly specified and relevant to the role.
Deduction Criteria:
- Missing critical skills or key technologies: -8 points
- Too few or overly general skills listed (e.g., "communication" without examples): -7 points
- No clear categorization of skills (hard vs. soft skills): -6 points
- Lack of technical specificity or out-of-date technologies: -9 points

8. Semantic Relevance [0-10]:
- Clear, professional terminology with appropriate industry-specific language.
- Coherent professional narrative (clear and concise job descriptions).
Deduction Criteria:
- Overuse of buzzwords or non-specific language: -8 points
- Lack of industry-specific terms or context: -9 points
- Incoherent, poorly structured job descriptions: -8 points
- Weak action verbs or vague descriptions: -7 points

9. Technical Compatibility [0-10]:
- Standard bullet points, no special characters, and consistent text alignment.
- No parsing issues that interfere with ATS reading the document.
Deduction Criteria:
- Special characters or formatting errors that disrupt parsing: -7 points
- Non-standard bullet points or misaligned sections: -8 points
- Hidden text or invisible sections causing ATS issues: -9 points
- Alignment inconsistencies that make parsing difficult: -6 points

10. Document Structure and Density [0-10]:
- Appropriate length based on the candidate's experience level (usually 1-2 pages).
- Balanced text-to-white space, clear section demarcations.
- Concise yet comprehensive content without irrelevant details.
Deduction Criteria:
- Overly long resume (more than 2 pages for most candidates): -9 points
- Sparse or excessively dense content with unclear section breaks: -8 points
- Missing or incomplete sections: -7 points
- Unnecessary content (e.g., irrelevant hobbies or outdated positions): -6 points

IMPORTANT:
- Provide ONLY the numerical score for each category.
- Use the exact category names:
  1. Keyword Matching
  2. Formatting and Readability
  3. File Format Compatibility
  4. Contact Information Clarity
  5. Work Experience Structure
  6. Education Details Precision
  7. Skills Section Comprehensiveness
  8. Semantic Relevance
  9. Technical Compatibility
  10. Document Structure and Density

Resume content:
{text}
"""


        
        try:
            model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)
            response = model.invoke(prompt_template.format(text=text))
            
            # Log the raw AI response for inspection
            logging.info(f"AI Response: {response.content}")
            
            # Check if the AI response is in a valid format (looking for "Aspect Name: Score")
            score_pattern = r"(\w[\w\s]*):\s*(\d{1,3})"  # Matches Aspect Name: Score
            matches = re.findall(score_pattern, response.content)
            
            if matches and len(matches) == 10:
                # Parse the scores for each category
                ats_scores = {}
                for match in matches:
                    aspect, score = match
                    ats_scores[aspect.strip()] = int(score)
                
                # Validate that all scores are within the range 0-100
                for aspect, score in ats_scores.items():
                    if score < 0 or score > 10:
                        raise ValueError(f"Invalid score for {aspect}: {score}")
                
                return ats_scores
            else:
                logging.warning(f"Expected 10 scores, but found {len(matches)} in the response: {response.content}")
                raise ValueError("Failed to extract valid scores from the AI response.")
        
        except Exception as e:
            logging.error(f"Error calculating ATS scores: {str(e)}")
            # Log the full traceback for better diagnosis
            logging.exception(e)
            
            # Return fallback scores in case of error
            return {
                "Keyword Matching": 0,
                "Formatting and Readability": 0,
                "File Format Compatibility": 0,
                "Contact Information Clarity": 0,
                "Work Experience Structure": 0,
                "Education Details Precision": 0,
                "Skills Section Comprehensiveness": 0,
                "Semantic Relevance": 0,
                "Technical Compatibility": 0,
                "Document Structure and Density": 0
                
            }



    def create_ats_chart(self, scores: dict):
        """
        Create a visually stunning presentation of ATS scores using radar and progress bar visualizations.
        
        Args:
            scores (dict): Dictionary containing ATS analysis scores
        
        Returns:
            plotly.graph_objects.Figure: Combined radar and progress bar chart
        """
        aspects = list(scores.keys())
        values = list(scores.values())
        
        # Calculate overall score (weighted average)
        weights = {
            "Keyword Matching": 0.15,
            "Formatting and Readability": 0.1,
            "File Format Compatibility": 0.1,
            "Contact Information Clarity": 0.05,
            "Work Experience Structure": 0.15,
            "Education Details Precision": 0.1,
            "Skills Section Comprehensiveness": 0.1,
            "Semantic Relevance": 0.1,
            "Technical Compatibility": 0.1,
            "Document Structure and Density": 0.05
        }
        
        overall_score = sum(scores[aspect] * weights[aspect]*10 for aspect in aspects)
        
        # Create radar chart
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the loop
            theta=aspects + [aspects[0]],  # Close the loop
            fill='toself',
            name='ATS Compatibility',
            line=dict(color='#3498db', width=2)
        ))
        
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10]),
                angularaxis=dict(visible=True)
            ),
            title="Radar Chart of ATS Scores",
            title_x=0.5,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        # Create progress bar for overall score
        progress_bar_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=overall_score,
            delta={'reference': 70},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "blue", 'width': 4},
                    'thickness': 0.75,
                    'value': overall_score
                }
            },
            title={'text': "Overall ATS Score"}
        ))
        
        progress_bar_fig.update_layout(
            margin=dict(t=30, b=30, l=30, r=30),
            height=400
        )
        
        return radar_fig, progress_bar_fig

    @staticmethod
    def extract_text_from_pdf(pdf_docs: List[tempfile._TemporaryFileWrapper]) -> str:
        try:
            text = ""
            for pdf in pdf_docs:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise ValueError("Failed to process PDF file")

    @staticmethod
    def create_text_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            texts = splitter.split_text(text)
            return [Document(page_content=t) for t in texts]
        except Exception as e:
            logger.error(f"Error creating text chunks: {str(e)}")
            raise ValueError("Failed to process text chunks")

    def create_vector_store(self, text_chunks: List[Document], save_path: str = "faiss_index") -> FAISS:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            texts = [doc.page_content for doc in text_chunks]
            vector_store = FAISS.from_texts(texts, embedding=embeddings)
            vector_store.save_local(save_path)
            return vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise ValueError("Failed to process resume content")

    def get_qa_chain(self) -> load_qa_chain:
        prompt_template = """
        You are a professional interview assistant. Create a warm, confident, and engaging introduction for a job interview scheduled with Harry Potter at 9:00 AM. The introduction should:

1. Begin with "Good morning, Harry Potter."
2. Flow conversationally, showcasing the candidate’s experience, key strengths, and recent achievements in a natural manner.
3. Relate the candidate’s expertise to the role they are interviewing for, highlighting why they are a great fit for the position.
4. Be succinct, keeping it under 60 seconds when spoken (approximately 150 words).
5. Sound authentic, with no overly formal or scripted language.

Base the introduction on this resume information: {context}

Your response should be warm, engaging, and confident—giving Harry Potter a clear sense of the candidate’s background and what they can bring to the role.
        """

        try:
            model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
            prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
            return load_qa_chain(model, chain_type="stuff", prompt=prompt)
        except Exception as e:
            logger.error(f"Error creating QA chain: {str(e)}")
            raise ValueError("Failed to initialize the AI model")

    def process_resume(self, text: str) -> str:
        try:
            docs = self.create_text_chunks(text)
            chain = self.get_qa_chain()
            response = chain({"input_documents": docs, "question": "Generate a professional introduction"})
            return response["output_text"]
        except Exception as e:
            logger.error(f"Error processing resume: {str(e)}")
            raise ValueError("Failed to process resume and generate introduction")
    def get_ats_improvement_suggestions(self, text: str) -> str:
        prompt_template = """
        You are an ATS (Applicant Tracking System) specialist. Based on the following resume content, provide detailed suggestions on how the candidate can improve their resume to increase compatibility with ATS systems and increase their chances of success in interviews.

        Focus on the following areas:
        1. Keyword optimization: Which important keywords are missing or underrepresented?
        2. Formatting improvements: How can the candidate improve resume formatting to make it more ATS-friendly?
        3. Skills coverage: Are there any crucial skills that the candidate should highlight more effectively?
        4. Education and experience: Does the resume clearly emphasize the candidate's education and experience in a way that aligns with job descriptions?

        Resume content:
        {context}

        Your response should be constructive and detailed, providing concrete steps the candidate can take to improve their resume.
        """

        try:
            model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
            prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            response = chain({"input_documents": [Document(page_content=text)], "question": "Provide ATS improvement suggestions"})
            return response["output_text"]
        except Exception as e:
            logger.error(f"Error generating ATS improvement suggestions: {str(e)}")
            raise ValueError("Failed to generate ATS improvement suggestions")

    def run(self):
        st.markdown("""
            <div class="header-container">
                <div class="header-text">Interview Introduction Generator and ATS Score Optimizer </div>
                <div class="subheader-text">Generate a warm, confident, and engaging introduction for a job interview</div>
            </div>
        """, unsafe_allow_html=True)

        upload_col1, upload_col2, upload_col3 = st.columns([1, 2, 1])
        with upload_col2:
            st.subheader("Upload Resume")
            pdf_docs = st.file_uploader(
                "Upload your resume (PDF)",
                accept_multiple_files=True,
                type=['pdf']
            )

            if st.button("Analyze", key="generate_btn", use_container_width=True):
                if not pdf_docs:
                    st.error("Please upload a resume first!")
                    return

                progress_bar = st.progress(0)  # Initialize the progress bar
                progress_text = st.empty()  # Placeholder for progress messages
                
                try:
                    progress_text.text("Extracting text from resume...")
                    raw_text = self.extract_text_from_pdf(pdf_docs)
                    st.session_state['raw_text'] = raw_text  # Add this line
                    progress_bar.progress(30)
                    
                    if not raw_text:
                        st.error("No text could be extracted from the PDF.")
                        progress_bar.empty()
                        return

                    progress_text.text("Analyzing ATS compatibility...")
                    ats_scores = self.calculate_ats_score(raw_text)
                    st.session_state['ats_scores'] = ats_scores
                    progress_bar.progress(60)
                    
                    progress_text.text("Generating professional introduction...")
                    introduction = self.process_resume(raw_text)
                    st.session_state['introduction'] = introduction
                    progress_bar.progress(80)  # Update to 80 after introduction generation

                    # Generate audio for the introduction
                    progress_text.text("Generating audio for the introduction...")
                    audio_file = self.audio_player.generate_speech(st.session_state['introduction'])
                    st.session_state['audio_file'] = audio_file
                    progress_bar.progress(90)  # Update to 90 after audio generation

                    # Generate ATS suggestions
                    progress_text.text("Generating ATS improvement suggestions...")
                    ats_improvement_suggestions = self.get_ats_improvement_suggestions(raw_text)
                    st.session_state['ats_improvement_suggestions'] = ats_improvement_suggestions
                    progress_bar.progress(100)  # Update to 100 after ATS suggestions
                    
                    st.session_state['show_results'] = True
                    progress_text.text("All done!")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logger.error(f"Application error: {str(e)}")
                    progress_bar.empty()
                    progress_text.empty()
                    return
                finally:
                    progress_bar.empty()
                    progress_text.empty()

        if 'show_results' in st.session_state and st.session_state['show_results']:
            results_col1, results_col2, results_col3 = st.columns([1, 6, 1])
            with results_col2:
                st.subheader("ATS Compatibility Score")
                radar_chart, progress_chart = self.create_ats_chart(st.session_state['ats_scores'])
                st.plotly_chart(progress_chart, use_container_width=True)
                st.plotly_chart(radar_chart, use_container_width=True)
                

                # Display average ATS score
                average_score = sum(st.session_state['ats_scores'].values()) / len(st.session_state['ats_scores'])
                        
                # Displaying Professional Introduction
                st.subheader("Your Professional Introduction")
                introduction_html = f"""
                <div class="introduction-container">
                    {st.session_state['introduction']}
                </div>
                    """
                st.markdown(introduction_html, unsafe_allow_html=True)


                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📋 Copy to Clipboard", use_container_width=True):
                        st.write("Introduction copied to clipboard!")
                        st.markdown(f"""
                            <script>
                                navigator.clipboard.writeText(`{st.session_state['introduction']}`);
                            </script>
                        """, unsafe_allow_html=True)

                with col2:
                    st.audio(st.session_state['audio_file'], format="audio/mp3")

                with col3:
                    st.download_button(
                        label="📥 Download Introduction",
                        data=st.session_state['introduction'],
                        file_name="professional_introduction.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                st.subheader("Suggestions for Improving Your Resume")
                ats_improvement_suggestions = st.session_state['ats_improvement_suggestions']
                st.markdown(ats_improvement_suggestions)

    
if __name__ == "__main__":
    app = InterviewIntroGenerator()
    app.run()