from crewai import Crew, Agent, Task, LLM

llm = LLM(
    model="groq/mixtral-8x7b-32768",
    temperature=0.7,
    api_key='YOUR_GROQ_API_KEY',
)

famous_people_finder = Agent(
    role="list down famous celibrities",
    goal="to search for famous people and list down all",
    backstory="You are a popular people finder, and your job is to:\n1. Find all the popular people like celbrities, sportsperson, politian, singers, etc\n2. List them down categorically",
    allow_delegation=False,
    llm=llm,
    verbose=True,
)

topic_related_info_finder = Agent(
    role="reaearching and finding information regarding topic",
    goal="to find information of famous_people gathered by famous_people_finder agent about the provided {topic}",
    backstory="You are a topic related information researcher, your job is to take the {topic} and famours people list and find information about that topic for each person ",
    allow_delegation=False,
    llm=llm,
    verbose=True,
)

similarity_searcher = Agent(
    role="finding a similarity about people with their information about the provided {topic}",
    goal="to find information of famous_people gathered by famous_people_finder agent about the provided {topic}",
    backstory="You area similarity searcher which have following jobs: 1. to get the information from topic_related_info_finder and run a similarity serach, 2. List these similarities with the names of the famous people",
    allow_delegation=False,
    llm=llm,
    verbose=True,
)

famous_people_find = Task(
    description = "find famous people",
    expected_output="list of all famous people ",
    llm=llm,
    agent = famous_people_finder,
    verbose=True,
)

find_topic_related_information = Task(
    description = "find the information about {topic} of the listed famous people",
    expected_output="JSON of {topic} related information about the listed famous people",
    llm=llm,
    agent = topic_related_info_finder,
    verbose=True,
)

similarity_search = Task(
    description="with the {topic} related information about famous people run a similarity search",
    expected_output="JSON of similarity between these famous people about {topic} related information about them",
    llm=llm,
    agent = similarity_searcher,
    verbose=True,
)

crew = Crew(
    agents=[famous_people_finder, topic_related_info_finder, similarity_searcher],
    tasks=[famous_people_find, find_topic_related_information, similarity_search],
    llm=llm,
    max_rpm=5,
    max_retries=3,
    max_concurrent_tasks=3,
    verbose=True,
)

result = crew.kickoff(inputs={"topic": "parents financial condition"})

