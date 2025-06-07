from langchain_ollama import ChatOllama, OllamaEmbeddings, OllamaLLM
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, ConversationChain
from pydantic import BaseModel, Field
from langchain.memory import ConversationBufferMemory


article = """
Llamas, native to the Andes Mountains of South America, have been domesticated for thousands of years. These intelligent and social animals belong to the camelid family, alongside alpacas and camels. Known for their thick wool and sturdy build, llamas have been used as pack animals, carrying loads up to 25–30% of their body weight across rugged terrains.  
Beyond their strength, llamas are also valued for their soft fleece, which is hypoallergenic and used in textiles. They communicate through a variety of sounds, including hums and alarm calls, and spit when annoyed—usually at other llamas.  
Today, llamas are found worldwide, serving as therapy animals, livestock guardians, and beloved pets. Their curious and friendly nature makes them great companions. Whether trekking through mountains or grazing on farms, llamas continue to be one of the most versatile and fascinating animals in the world.
"""

system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an AI assistant that helps generate article titles."
)

first_user_prompt = HumanMessagePromptTemplate.from_template(
    template="""You are tasked with creating a name for a article.
The article is here for you to examine {article}

Generate 3 unique names for the article. 
The names should be based of the context of the article. 
Be creative, but make sure the names are clear, catchy, 
and relevant to the theme of the context.

Compare all the names, and decide which name is best based on
How catchy the name is, How creative the name is, and how relevant the name is.

only output the best name as:
Article Name: ...""",
    input_variables=["article"],
)

second_user_prompt = HumanMessagePromptTemplate.from_template(
    template="""You are tasked with creating a description for
the article. The article is here for you to examine:

---

{article}

---

Here is the article title '{article_title}'.

Output the SEO friendly article description. Do not output
anything other than the description.""",
    input_variables=["article", "article_title"],
)

third_user_prompt = HumanMessagePromptTemplate.from_template(
    template="""You are tasked with creating a new paragraph for the
article. The article is here for you to examine:

---

{article}

---

Choose one paragraph to review and edit. During your edit,
ensure you provide constructive feedback to the user so they
can learn where to improve their own writing.""",
    input_variables=["article"],
)


class Paragraph(BaseModel):
    original_paragraph: str = Field(
        description="The original paragraph"
    )
    edited_paragraph: str = Field(
        description="The improved edited paragraph"
    )
    feedback: str = Field(
        description="Constructive feedback on the original paragraph"
    )


def main_article():
    model_name = "qwen2.5:7b"
    llm = ChatOllama(temperature=0.0, model=model_name)
    structured_llm_Paragraph = llm.with_structured_output(Paragraph)

    first_prompt = ChatPromptTemplate.from_messages([
        system_prompt,
        first_user_prompt
    ])
    # print(first_prompt.format(article="TEST STRING"))

    chain_one = (
        {"article": lambda x: x["article"]}
        | first_prompt
        | llm
        | {"article_title": lambda x: x.content}
    )

    res = chain_one.invoke({"article": article})
    print(res)

    second_prompt = ChatPromptTemplate.from_messages([
        system_prompt,
        second_user_prompt
    ])

    chain_two = (
        {
            "article": lambda x: x["article"],
            "article_title": lambda x: x["article_title"]
        }
        | second_prompt
        | llm
        | {"summary": lambda x: x.content}
    )

    res = chain_two.invoke({
        "article": article,
        "article_title": res["article_title"]
    })

    print(res)

    third_prompt = ChatPromptTemplate.from_messages([
         system_prompt,
         third_user_prompt
    ])

    chain_three = (
        {"article": lambda x: x["article"]}
        | third_prompt
        | structured_llm_Paragraph
        | {
            "original_paragraph": lambda x: x.original_paragraph,
            "edited_paragraph": lambda x: x.edited_paragraph,
            "feedback": lambda x: x.feedback
        }
    )

    res = chain_three.invoke({"article": article})
    print(res)


def main():
    model_name = "qwen2.5:7b"
    llm = ChatOllama(temperature=0.0, model=model_name)

    memory = ConversationBufferMemory(return_messages=True)

    memory.chat_memory.add_user_message("Hi, my name is Josh")
    memory.chat_memory.add_ai_message("Hey Josh, what's up? I'm an AI model called Zeta.")
    memory.chat_memory.add_user_message("I'm researching the different types of conversational memory.")
    memory.chat_memory.add_ai_message("That's interesting, what are some examples?")

    x = memory.load_memory_variables({})
    print(x)

    chain = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

    res = chain.invoke({"input": "what is my name again?"})
    print(res)



if __name__ == "__main__":
    # main_article()
    main()
