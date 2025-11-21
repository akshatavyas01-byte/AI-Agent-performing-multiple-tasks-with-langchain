# AI-Agent-performing-multiple-tasks-with-langchain

### Overview 
This project implements a smart **Ai-powered Json report generation of an Act  ** using **LangChain**.
It allows users to perform 4 tasks:

#### Upload and extraction
-Upload their PDF, extract clean full text of the pdf.

#### Summarization
-Summarize the text focusing on the specific topics with bullet points.
-Topics:- Purpose, Key definitions, Eligibility, Obligations, Enforcement elements 

####Extract Key Legislative Sections 
-Extract key sectioins from the text.
-Sections:- Definitions, Obligations, Responsibilities, Eligibility, Payments / Entitlements, Penalties / Enforcement, Record-keeping / Reporting

#### Rule Checking 
-Checks if the following rules are being passed.
-Rules:
1. Act must define key terms 
2. Act must specify eligibility criteria 
3. Act must specify responsibilities of the administering authority 
4. Act must include enforcement or penalties 
5. Act must include payment calculation or entitlement structure 
6. Act must include record-keeping or reporting requirements

### Json report
- Generates a full Json report of the Act consisting the above tasks.

---
## Features
1. Extracts the text from the pdf.
2. Build chains for the each of the sepcific tasks such as summarization, Key Extraction and Rule checking.
3. Integrates **openai/gpt-oss-20b** LLM model from HuggingFace in these chains.
4. Executes all the task chains in parallel.
5. Return a stuctured full json report.


## Architecture
PDF Document-> Extraction of text ->  Parallel Chains(Summary_chain, Key_Extraction_chain, Rule_Checking_chain) -> Json Report


Summary_chain => prompt -> llm -> JsonOutputParser

Key_Extraction_chain => prompt -> llm -> JsonOutputParser

Rule_Checking_chain => prompt -> llm -> JsonOutputParser

```python
# 1. Install dependencies
!pip install -q -U langchain langchain-core langchain-community langchain-openai langchain-huggingface
!pip install pypdf

# 2. Step-by Step Implementation
# 1. A Runnable Lambda Fuction for the PDF loading and Text Extraction from the pdf:
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
def Get_pdf_text(path:dict)->str:
  if path!=None:
    pdf_url=path['input']
  else:
    pdf_url='/content/ukpga_20250022_en.pdf'
  pdf_loader=PyPDFLoader(pdf_url)
  pages=pdf_loader.load()
  full_text=''
  for p in pages:
    full_text+=p.page_content
  return full_text

Get_text=RunnableLambda(Get_pdf_text)

# 2. Load Api Keys from google collab:
from google.colab import userdata
api_key=userdata.get('PROJECT')

# 3. Intialization of HuggingFace Model:
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
llm=HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    huggingfacehub_api_token=api_key,
    temperature=0.2
        )
llm_chat=ChatHuggingFace(llm=llm)

# 4. Summary Chain:
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

summary_template='''
Your a halpful assistance.
Task:Summarize the given Act context.

Instructions:
1.Focus on the following points in the summary:
- Purpose
- Key definitions
- Eligibility
- Obligations
- Enforcement elements
2. Pay attention that the summary should not be less than 5 bullet points and not exceed more than 10 bullet points.

Fromat:Json.
Example:{{
  'summary_points':'['1.Point1', '2.Point2'.......'Point10']'}}
  
Context:
{context}
'''
summary_prompt=PromptTemplate(template=summary_template)

summary_chain=(
    {"context": lambda x: x}
    | summary_prompt
    | llm_chat
    | JsonOutputParser()
)

# 5. Key Extraction Chain:

KeyExtractions_template='''
Your a helpful assitant.
Task: Extract key Legislative Sections from the given Act context.

Instructions:
1. The following Key should be extracted:
- Definitions
- Obligations
- Responsibilities
- Eligibility
- Payments / Entitlements
- Penalties / Enforcement
- Record-keeping / Reporting

2. Pay attention the result should be strictly in the Json format.

Fromat: Json with the following keys:
-definations
-obligations
-responsibilities
-eligibility
-payments
-panalties
-record_keeping

Context:
{context}
'''
KeyExtractions_prompt=PromptTemplate(template=KeyExtractions_template)

KeyExtraction_chain=(
    {'context': lambda x:x}
    | KeyExtractions_prompt
    | llm_chat
    | JsonOutputParser()
)

#6. Rule Check Chain:

RuleCheck_template='''
Your a smart and helpful assistant.
Task:Appy the following rule checks on the given Act context.
Instructions:
1. The following rules should be checked:
-Act must define key terms
-Act must specify eligibility criteria
-Act must specify responsibilities of the administering authority
-Act must include enforcement or penalties
-Act must include payment calculation or entitlement structure
-Act must include record-keeping or reporting requirements

2. Pay attention the result should be strictly in the Json format.

Fromat: For each rule check the Json should consist the following keys:
-rule
-status
-evidence
-confidence

Example for one such rule Check is:
{{
 "rule": "Act must define key terms",
 "status": "pass",
 "evidence": "Section 2 – Definitions",
 "confidence": 92

}}

Act Context:
{Context}
'''

RuleCheck_prompt=PromptTemplate(template=RuleCheck_template)

RuleCheck_chain=(
    {'Context': lambda x:x}
    | RuleCheck_prompt
    | llm_chat
    | JsonOutputParser()
)


# 7. Parallel Chains:
from langchain_core.runnables import RunnableParallel

parallel_chains=RunnableParallel(
    summary=summary_chain,
    key_extraction=KeyExtraction_chain,
    rule_check=RuleCheck_chain
)

# 8. Full chain:
 Full_chain=(
    Get_text
    | parallel_chains
)

# 9. Run the Full Chain:
Full_chain.invoke({'input':'/content/ukpga_20250022_en.pdf'})

```

Author
Akshata Vyas









































