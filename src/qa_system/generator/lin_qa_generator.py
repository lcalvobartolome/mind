import logging
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain
from gptcache.adapter.langchain_models import LangChainChat
from gptcache import cache
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

class LinQAGenerator(object):
    def __init__(
        self,
        open_api_key: str,
        temperature: float = 0,
        logger: logging.Logger = None):
        
        self._logger = logger if logger else logging.getLogger(__name__)
        
        # Set OPENAI API KEY
        os.environ["OPENAI_API_KEY"] = open_api_key
        
        # get the content(only question) form the prompt to cache
        def get_msg_func(data, **_):
            return data.get("messages")[-1].content
        
        onnx = Onnx()
        cache_base = CacheBase('sqlite')
        vector_base = VectorBase('faiss', dimension=onnx.dimension)
        data_manager = get_data_manager(cache_base, vector_base)
        cache.init(
            pre_embedding_func=get_msg_func,
            embedding_func=onnx.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(),
            )
        cache.set_openai_key()
        

        chat = LangChainChat(chat=ChatOpenAI(temperature=temperature))
        
        # Create generator
        self.generator = QAGenerationChain.from_llm(chat)

    
    def generate_qa_pairs(self, data_path):
        
        # Create loader
        loader = CSVLoader(
            file_path = data_path,
            source_column = "doc_id",
            csv_args = {
                "fieldnames": ["doc_id", "text", "url"]
            }
        )
        
        data = loader.load()
        