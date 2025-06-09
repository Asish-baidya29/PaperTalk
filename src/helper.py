import os
import importlib
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS 
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader




class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    



class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec
    




# Load env variables
load_dotenv()
LLAMA2_PI_KEY=os.getenv("LLAMA2_PI_KEY")
os. environ['LLAMA2_PI_KEY']=LLAMA2_PI_KEY


# PDF text extraction 
from io import BytesIO

def extract_PDF_texts(uploaded_file):
    if uploaded_file is not None:
        pdf_stream = BytesIO(uploaded_file.read())  # Wrap bytes in BytesIO
        pdf_reader = PdfReader(pdf_stream)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    return ""


# chunking
def text_chunks(text, chunk_size=1500, overlap=100):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size = chunk_size,chunk_overlap=overlap)
    chunked_txt= text_splitter.split_text(text)
    return chunked_txt

def get_vector_store(chunked_txt):
    embeddings=HuggingFaceEmbeddings()
    vector_store=FAISS.from_texts(chunked_txt,embedding=embeddings)
    return vector_store 




"""# DataLoader for GPT

def create_dataloader_v1(text, batch_size=4, max_length=200, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
     # Check for empty input
    if not text or len(text.strip()) == 0:
        raise ValueError("Input text is empty. Cannot create DataLoader.")
    
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    # Tokenize and check token count
    token_ids = tokenizer.encode(text)
    if len(token_ids) < max_length:
        raise ValueError(f"Not enough tokens to create sequences: got {len(token_ids)}, need at least {max_length}.")
    # Create dataset
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    if len(dataset) == 0:
        raise ValueError("Dataset has zero samples. Check your text length or tokenizer config.")
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

# embedding
def embedding(dataloader):

    # token embedding
    data_iter = iter(dataloader)
    input_batch, target_batch = next(data_iter)
    # GPT-2 vocab size
    vocab_size = 50257 
    # typical embedding size 
    embedding_dim = 768  
    torch.manual_seed(123)
    token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
    token_embeddings = token_embedding_layer(input_batch)

    #positional embedding
    max_length = 200
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length,embedding_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    
    input_embeddings = token_embeddings + pos_embeddings

    return input_embeddings


# casual attention 

def self_attention (input_embeddings):
    d_in = input_embeddings.shape[2] #B 768
    d_out = 500 #C whatever i want based on model
    seq_len = 10 
    input_embeddings = torch.randn(seq_len, d_in)  # Shape: (seq_len, d_in)
    batch = torch.stack((input_embeddings, input_embeddings), dim=0)

    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length=context_length, dropout=0.1)
    context_vecs = ca(batch)   
    return context_vecs"""
            
# vector db
from langchain.docstore.document import Document

def vector_db(context_vecs):
    # Ensure context_vecs is a list of float32 numpy arrays
    if isinstance(context_vecs, torch.Tensor):
        List_context_vecs = context_vecs.detach().cpu().numpy().astype("float32")
   
    # Wrap texts in Document format
    #documents = [Document(page_content=txt) for txt in text]

    # Create FAISS store with precomputed vectors
    vector_store = FAISS(embedding_function=None, index=None, docstore=None)
    vector_store.index = FAISS.IndexFlatL2(context_vecs.shape[1])
    vector_store.index.add(context_vecs)
    #vector_store.docstore.add(documents)

    return vector_store

# memory
def conversational_chain(vector_store):
    model_id = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return conversation_chain

