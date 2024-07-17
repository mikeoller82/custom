import random
import string
import streamlit as st
from langgraph_sdk import get_client
import asyncio
from langsmith import Client
from streamlit_extras.stylable_container import stylable_container
from copy import deepcopy

st.set_page_config(layout="wide")

# Langsmith feedback client
feedback_client = Client(api_url="https://beta.api.smith.langchain.com")

# Find run id for giving feedback
async def get_run_id_corresponding_to_node(client, thread, node_id):
    '''Get the run id corresponding to the task executed'''
    runs = await client.runs.list(thread_id=thread['thread_id'])
    for r in runs:
        if r['kwargs']['config']['configurable']['node_id'] == node_id:
            return r['run_id']
    return None

# Create the agent
async def start_agent(session_id):
    client = get_client(url="https://your-langgraph-cloud-url.com")
    assistants = await client.assistants.search()
    assistants = [a for a in assistants if not a['config']]
    thread = await client.threads.create(metadata={"user":session_id})
    assistant = assistants[0]
    await asyncio.sleep(5.5)
    return [client, thread, assistant]

# Find a task that we had previously executed
async def get_thread_state(client, thread_id):
    return await client.threads.get_state(thread_id)

# Find tasks user has executed
async def get_user_threads(client, session_id):
    threads = await client.threads.search(metadata={"user":session_id})
    untitled_count = 1
    for t in threads:
        t_state = await get_thread_state(client, t['thread_id'])
        try:
            t['task_title'] = t_state['values']['task_title']
        except:
            t['task_title'] = f"Untitled task #{untitled_count}"
            untitled_count += 1
    return threads

llm_to_title = {
    "starting": "Waiting for user input ...",
    "web_search_llm": "Searching the web...",
    "planner_llm": "Planning task execution...",
    "discovery_llm": "Discovering insights...",
    "replanner_llm": "Refining the plan...",
    "final_response_llm": "Generating final response..."
}

# Streaming task execution
async def generate_answer(placeholder, placeholder_title, input, client, thread, assistant, metadata = {}):
    current_llm = "starting"
    placeholder_title.markdown(f"<h4 style='text-align: center; color: rgb(206,234,253);'>{llm_to_title[current_llm]}</h4>", unsafe_allow_html=True)
    current_ind = 0
    ans = ""
    async for chunk in client.runs.stream(
        thread['thread_id'], assistant['assistant_id'], input=input, config={"configurable":metadata},
        stream_mode="messages", multitask_strategy="rollback"
    ):
        if chunk.data and 'run_id' not in chunk.data:
            if isinstance(chunk.data, dict):
                try:
                    current_llm = chunk.data[list(chunk.data.keys())[0]]['metadata']['name']
                    placeholder_title.markdown(f"<h4 style='text-align: center; color: rgb(206,234,253);'>{llm_to_title[current_llm]}</h4>", unsafe_allow_html=True)
                except:
                    pass
            elif current_llm == "final_response_llm" and chunk.data[0]['content']:
                ans += chunk.data[0]['content'][current_ind:]
                placeholder.info(ans)
                current_ind += len(chunk.data[0]['content'][current_ind:])

# Update variables after task has been executed
async def get_current_state(client, thread):
    current_state = await client.threads.get_state(thread_id=thread['thread_id'])
    return current_state

# When user selects a different task to view
async def update_current_state(client, thread, values):
    await client.threads.update_state(thread_id=thread['thread_id'], values=values)

# Create new thread for new task
async def get_new_thread(client, session_id):
    thread = await client.threads.create(metadata={"user":session_id})
    return thread

# Make sure event loop never closes
async def call_async_function_safely(func, *args):
    try:
        result = await func(*args)
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = await func(*args)
        else:
            raise e
    return result

# Helper function for task options
def transform_titles_into_options(titles):
    name_counts = {}
    for name in set(titles):
        name_counts[name] = titles.count(name)
    transformed_titles = []
    for name in titles[::-1]:
        count = name_counts[name]
        if count > 1 or titles.count(name) > 1:
            transformed_titles.append(f"{name} #{count}")
        else:
            transformed_titles.append(name)
        name_counts[name] -= 1
    return transformed_titles[::-1]

# Update variables after execution
async def update_session_variables():
    current_state = await call_async_function_safely(get_current_state, st.session_state.client, st.session_state.thread)
    st.session_state.task_graph = current_state['values']['task_graph']
    st.session_state.task_title = current_state['values']['task_title']
    st.session_state.currently_selected_task = str(current_state['values']['task_id_viewing'])
    st.session_state.current_node_id = str(int(current_state['values']['current_task_id']) + 1)

# Reset variables on new task
async def reset_session_variables():
    st.session_state.task_graph = {"-1": {'content': "Click Start Task to begin!", 'title': "Pre-start Task"}}
    st.session_state.currently_selected_task = "-1"
    st.session_state.task_number = 0
    st.session_state.current_node_id = '1'
    st.session_state.task_title = ""

async def stream(*args):
    await asyncio.gather(call_async_function_safely(generate_answer, *args))

async def main():
    st.markdown("""
    <style>
    .centered-title {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if "task_title" not in st.session_state or st.session_state.task_title == "":
        st.markdown("<h1 class='centered-title'>Task Resolving with LangGraph</h1>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h1 class='centered-title'>{st.session_state.task_title}</h1>", unsafe_allow_html=True)

    if "page_loaded" not in st.session_state:
        st.session_state.page_loaded = False
        st.session_state.task_title = ""
        st.session_state.num_selected = 0
        st.session_state.processing = False

    if "session_id" not in st.session_state:
        st.session_state.session_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

    if st.session_state.page_loaded == False:
        st.session_state.client, st.session_state.thread, st.session_state.assistant = await call_async_function_safely(start_agent, st.session_state.session_id)
        await reset_session_variables()
        st.session_state.page_loaded = True

    if "task_started" not in st.session_state:
        st.session_state.task_started = False

    if "show_start_input" not in st.session_state:
        st.session_state.show_start_input = False

    if "show_load_task" not in st.session_state:
        st.session_state.show_load_task = False

    if 'start_submit' in st.session_state and st.session_state.start_submit == True:
        st.session_state.running = True
    else:
        st.session_state.running = False

    # Starting/New task
    if st.session_state.show_start_input:
        task_text = st.sidebar.text_area("Task Description", disabled=st.session_state.running)
        col1, col2 = st.sidebar.columns([1, 1]) 
        with col1:
            if st.button("Back", key="start-back", disabled=st.session_state.running):
                st.session_state.show_start_input = False
                st.session_state.processing = False
                st.rerun()
        with col2:
            if st.button("Submit", key="start_submit", disabled=st.session_state.running):
                if st.session_state.task_started:
                    st.session_state.thread = await call_async_function_safely(get_new_thread, st.session_state.client, st.session_state.session_id)
                    await reset_session_variables()     
                await stream(st.session_state.box, st.session_state.box_title, {'task': task_text}, st.session_state.client, st.session_state.thread,
                             st.session_state.assistant, {"node_id": st.session_state.current_node_id})
                st.session_state.task_started = True
                await update_session_variables()
                st.session_state.show_start_input = False
                st.session_state.processing = False
                st.session_state.task_number = 1
                st.rerun()

    # Loading task
    elif st.session_state.show_load_task:
        col1, col2 = st.sidebar.columns([1, 1])
        threads = await call_async_function_safely(get_user_threads, st.session_state.client, st.session_state.session_id)
        threads_without_current = [t for t in threads if t['thread_id'] != st.session_state.thread['thread_id'] and 'Untitled' not in t['task_title']]
        options = [t['task_title'] for t in threads_without_current]

        if len(options) > 0:
            selected_task = st.sidebar.selectbox("", options, index=None, placeholder="Select task", 
                                                 label_visibility="collapsed", key=f"task_selector")
        else:
            selected_task = None
            st.sidebar.write("No alternate tasks!")

        if selected_task is not None:
            st.session_state.thread = threads_without_current[options.index(selected_task)]
            await update_session_variables()
            st.session_state.show_load_task = False
            st.session_state.task_number = 1
            st.rerun()
        with col1:
            if st.button("Back", key="load-task-back"):
                st.session_state.show_load_task = False
                st.rerun()
    # Default Navigation pane
    else:
        st.sidebar.header("Navigation")
        if st.sidebar.button("New Task" if st.session_state.task_started else "Start Task"):
            st.session_state.processing = True
            st.session_state.show_start_input = True
            st.rerun()
        elif st.sidebar.button("Load Task"):
            st.session_state.show_load_task = True
            st.rerun()

    st.sidebar.write(" ")  

    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] > * {
        max-height: 300px;
        overflow-y: scroll;
    }
    </style>
    """, unsafe_allow_html=True)

    _, col_middle_title, _ = st.columns([1, 6, 1])
    
    if "box_title" not in st.session_state or st.session_state.processing == False:
        st.session_state.box_title = col_middle_title.empty()
    elif st.session_state.processing == True:
        with col_middle_title:
            st.markdown(f"<h4 style='text-align: center; color: rgb(206,234,253);'>Waiting for user input...</h4>", unsafe_allow_html=True)

    st.session_state.task_title = st.markdown(f"<h2 style='text-align: center; color: white;'>{st.session_state.task_graph[st.session_state.currently_selected_task]['title']}</h2>", unsafe_allow_html=True)
    _, col_middle, col_scroll = st.columns([1, 6, 1])
    if "box" not in st.session_state:
        st.session_state.box = col_middle.empty()
        st.rerun()
    else:
        st.session_state.box.info(st.session_state.task_graph[st.session_state.currently_selected_task]['content'])

    with col_scroll:
        st.write("🔺\nScroll\n🔻")
    
    st.markdown(f"<h5 style='text-align: center;'>Task {st.session_state.task_number}</h5>", unsafe_allow_html=True)

    _, col2, _ = st.columns([1, 2, 1])
    if st.session_state.processing == False:
        # Feedback options
        if st.session_state.current_node_id != '1':
            _, col2a, col2b, _ = st.columns([1, 1, 1, 1])
            # Bad result
            with col2a:
                with stylable_container(
                    key="red_button",
                    css_styles="""
                        button {
                            background-color: red;
                            color: white;
                            border-radius: 20px;
                            margin-left: 80px;
                        }
                        """,
                ):
                    if st.button("Bad result", key="red_button"):
                        run_id = await call_async_function_safely(get_run_id_corresponding_to_node, st.session_state.client,
                                                                  st.session_state.thread, str(int(st.session_state.current_node_id)-1))
                        feedback_client.create_feedback(
                            run_id=run_id,
                            key="feedback-key",
                            score=0.0,
                            comment="Bad result",
                        )
            # Good result
            with col2b:
                with stylable_container(
                    key="green_button",
                    css_styles="""
                        button {
                            background-color: green;
                            color: white;
                            border-radius: 20px;
                            margin-left: 80px;
                        }
                        """,
                ):
                    if st.button("Good result", key="green_button"):
                        run_id = await call_async_function_safely(get_run_id_corresponding_to_node, st.session_state.client,
                                                                  st.session_state.thread, str(int(st.session_state.current_node_id)-1))
                        feedback_client.create_feedback(
                            run_id=run_id,
                            key="feedback-key",
                            score=1.0,
                            comment="Good result",
                        )

if __name__ == "__main__":
    asyncio.run(main())