from pypdf import PdfReader
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer



def preprocess_text(text):
    # Remove email, URLs, and special characters (except alphanumeric and spaces)
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters except words and numbers

    # Replace multiple spaces and newlines with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()

    return text


# creating a pdf reader object
reader = PdfReader('res2.pdf')

# printing number of pages in pdf file
print(len(reader.pages))

# getting a specific page from the pdf file
page = reader.pages[0]

# extracting text from page
cvData = page.extract_text()
print(cvData)


cvData = preprocess_text(cvData)
print(cvData)

jobData =  '''
We are seeking a talented Mobile App Developer Intern to assist in developing an innovative EdTech platform. This app will feature tools like Application Filling, Application Tracking, AI Profile Matcher, and Scholarship Finder to streamline the study abroad process for students. This is a great opportunity to work with cutting-edge technologies and contribute to a meaningful project.

Key Responsibilities:
1. Develop and maintain a cross-platform mobile app using frameworks like Flutter or React Native.
2. Implement key features such as Application Filling, Tracking, AI Profile Matching, and Scholarship Finder.
3. Integrate APIs for backend communication and data processing.
4. Optimize app performance for speed, responsiveness, and user experience.
5. Collaborate with the team to design intuitive UI/UX based on user needs.
6. Debug and troubleshoot issues to ensure smooth functionality.
7. Stay updated on the latest mobile development trends and tools.

Skills and Qualifications:
1. Experience with Flutter or React Native for cross-platform development.
2. Knowledge of RESTful APIs and backend integration.
3. Familiarity with UI/UX principles and tools like Figma or Adobe XD.
4. Basic understanding of mobile app performance optimization.
5. Strong problem-solving skills and attention to detail.
6. Eagerness to learn and adapt to new technologies.
'''


cvData = preprocess_text(cvData)
jobData = preprocess_text(jobData)


model = SentenceTransformer('all-MiniLM-L6-v2')
cvembeddings = model.encode(cvData)
jobembeddings = model.encode(jobData)

def calculate_similarity(jd_embedding, resume_embedding):
    return cosine_similarity([jd_embedding], [resume_embedding])[0][0]

similarity_score = calculate_similarity(jobembeddings, cvembeddings)
print("match score = " , similarity_score)
