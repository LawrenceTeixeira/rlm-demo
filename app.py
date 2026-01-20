"""
Streamlit app to demonstrate RLM (Reasoning Language Model) with REPL environment.
"""

import streamlit as st
import sys
import io
from contextlib import contextmanager
from typing import List, Dict, Optional
import time

# Import RLM components
from rlm.rlm_repl import RLM_REPL
from rlm.repl import REPLEnv
from rlm.utils.llm import OpenAIClient
from rlm.utils.prompts import DEFAULT_QUERY, next_action_prompt, build_system_prompt
import rlm.utils.utils as utils


class StreamlitLogger:
    """Logger that captures output for Streamlit display using expanders for better fit."""
    
    def __init__(self, output_container, status_placeholder):
        self.output_container = output_container
        self.status_placeholder = status_placeholder
        self.steps = []  # List of step data for rendering
        self.conversation_step = 0
        self.query = ""
    
    def log_query_start(self, query: str):
        """Log the start of a query."""
        self.query = query
        self.status_placeholder.info(f"🚀 **Query:** {query}")
    
    def log_initial_messages(self, messages: List[Dict[str, str]]):
        """Log initial system messages in a collapsed expander."""
        content = ""
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            msg_content = msg.get('content', '')
            if len(msg_content) > 300:
                msg_content = msg_content[:300] + "..."
            content += f"**[{i+1}] {role.upper()}:**\n{msg_content}\n\n"
        
        self.steps.append({
            "type": "system",
            "title": "📋 System Prompt",
            "content": content,
            "expanded": False
        })
        self._render()
    
    def log_model_response(self, response: str, has_tool_calls: bool):
        """Log model response."""
        self.conversation_step += 1
        
        display_response = response
        if len(response) > 800:
            display_response = response[:800] + "\n\n*... (truncated)*"
        
        status = "⚙️ Executing REPL code..." if has_tool_calls else "✅ Checking for final answer..."
        self.status_placeholder.info(f"Step {self.conversation_step}: {status}")
        
        self.steps.append({
            "type": "response",
            "title": f"🤖 Step {self.conversation_step}: Model Response",
            "content": display_response,
            "has_code": has_tool_calls,
            "expanded": True
        })
        self._render()
    
    def log_code_execution(self, code: str, stdout: str, stderr: str, execution_time: float):
        """Log code execution with syntax highlighting."""
        # Truncate for display
        display_code = code if len(code) <= 500 else code[:500] + "\n# ... (truncated)"
        display_stdout = stdout if len(stdout) <= 300 else stdout[:300] + "\n... (truncated)"
        
        output_section = ""
        if stdout:
            output_section += f"**Output:**\n```\n{display_stdout}\n```\n"
        if stderr:
            output_section += f"**Error:**\n```\n{stderr[:200]}\n```\n"
        if execution_time:
            output_section += f"\n*⏱️ {execution_time:.3f}s*"
        
        self.steps.append({
            "type": "code",
            "title": f"💻 Code Execution",
            "code": display_code,
            "output": output_section,
            "expanded": True
        })
        self._render()
    
    def log_final_response(self, response: str):
        """Log the final response prominently."""
        self.status_placeholder.success("🎯 **Final answer found!**")
        self.steps.append({
            "type": "final",
            "title": "🎯 FINAL ANSWER",
            "content": response,
            "expanded": True
        })
        self._render()
    
    def _render(self):
        """Render all steps using expanders for compact display."""
        with self.output_container:
            # Clear and re-render
            for i, step in enumerate(self.steps):
                step_type = step["type"]
                
                if step_type == "system":
                    with st.expander(step["title"], expanded=step["expanded"]):
                        st.markdown(step["content"], unsafe_allow_html=True)
                
                elif step_type == "response":
                    with st.expander(step["title"], expanded=step["expanded"]):
                        st.markdown(step["content"])
                        if step.get("has_code"):
                            st.caption("⚙️ Contains REPL code - executing...")
                
                elif step_type == "code":
                    with st.expander(step["title"], expanded=step["expanded"]):
                        st.code(step["code"], language="python")
                        if step["output"]:
                            st.markdown(step["output"])
                
                elif step_type == "final":
                    st.success(f"### {step['title']}")
                    st.markdown(step["content"])


class StreamlitRLM_REPL(RLM_REPL):
    """RLM_REPL with Streamlit-compatible logging."""
    
    def __init__(self, 
                 streamlit_logger: StreamlitLogger,
                 api_key: Optional[str] = None, 
                 model: str = "gpt-5",
                 recursive_model: str = "gpt-5-nano",
                 max_iterations: int = 10):
        self.api_key = api_key
        self.model = model
        self.recursive_model = recursive_model
        self.llm = OpenAIClient(api_key, model)
        
        self.repl_env = None
        self.depth = 0
        self._max_iterations = max_iterations
        
        # Use Streamlit logger
        self.st_logger = streamlit_logger
        
        self.messages = []
        self.query = None
    
    def setup_context(self, context: List[str] | str | List[Dict[str, str]], query: Optional[str] = None):
        if query is None:
            query = DEFAULT_QUERY

        self.query = query
        self.st_logger.log_query_start(query)

        self.messages = build_system_prompt()
        self.st_logger.log_initial_messages(self.messages)
        
        context_data, context_str = utils.convert_context_for_repl(context)
        
        self.repl_env = REPLEnv(
            context_json=context_data, 
            context_str=context_str, 
            recursive_model=self.recursive_model,
        )
        
        return self.messages

    def completion(self, context: List[str] | str | List[Dict[str, str]], query: Optional[str] = None) -> str:
        self.messages = self.setup_context(context, query)
        
        for iteration in range(self._max_iterations):
            response = self.llm.completion(self.messages + [next_action_prompt(query, iteration)])
            
            code_blocks = utils.find_code_blocks(response)
            self.st_logger.log_model_response(response, has_tool_calls=code_blocks is not None)
            
            if code_blocks is not None:
                self.messages = self._process_code_with_logging(
                    response, self.messages, code_blocks
                )
            else:
                assistant_message = {"role": "assistant", "content": "You responded with:\n" + response}
                self.messages.append(assistant_message)
            
            final_answer = utils.check_for_final_answer(response, self.repl_env, None)

            if final_answer:
                self.st_logger.log_final_response(final_answer)
                return final_answer
            
        self.messages.append(next_action_prompt(query, iteration, final_answer=True))
        final_answer = self.llm.completion(self.messages)
        self.st_logger.log_final_response(final_answer)
        return final_answer
    
    def _process_code_with_logging(self, response: str, messages: List[Dict], code_blocks: List[str]) -> List[Dict]:
        """Process code execution and log to Streamlit."""
        assistant_message = {"role": "assistant", "content": response}
        messages.append(assistant_message)
        
        for code_block in code_blocks:
            result = self.repl_env.code_execution(code_block)
            
            # Log the execution
            self.st_logger.log_code_execution(
                code_block, 
                result.stdout, 
                result.stderr, 
                result.execution_time
            )
            
            # Truncate output for context window
            truncated_stdout = result.stdout[:2000] if len(result.stdout) > 2000 else result.stdout
            truncated_stderr = result.stderr[:500] if len(result.stderr) > 500 else result.stderr
            
            tool_response = {
                "role": "user",
                "content": f"REPL environment returned:\nstdout: {truncated_stdout}\nstderr: {truncated_stderr}"
            }
            messages.append(tool_response)
        
        return messages


def main():
    st.set_page_config(
        page_title="RLM Demo by Lawrence Teixeira",
        page_icon="🧠",
        layout="wide"
    )
    
    # Title at the very top
    st.title("🧠 RLM Demo by Lawrence Teixeira")
    st.caption("Watch how RLM uses a REPL environment to recursively process contexts with sub-LLMs")
    
    # Custom CSS for better fit and reduce top padding
    st.markdown("""
    <style>
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    .stExpander > div:first-child {
        padding: 8px 12px;
    }
    div[data-testid="stVerticalBlock"] > div {
        padding-top: 0.5rem;
    }
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 1rem;
    }
    header[data-testid="stHeader"] {
        height: 0rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model = st.selectbox(
            "Root Model",
            ["gpt-5", "gpt-4o", "gpt-4o-mini"],
            index=0
        )
        
        recursive_model = st.selectbox(
            "Sub-LLM Model",
            ["gpt-5-nano", "gpt-4o-mini", "gpt-4o"],
            index=0
        )
        
        max_iterations = st.slider(
            "Max Iterations",
            min_value=1,
            max_value=20,
            value=10
        )
        
        st.subheader("📋 Example Settings")
        
        example_size = st.select_slider(
            "Example Size (lines)",
            options=[100, 1000, 10000, 100000, 1000000],
            value=1000,
            format_func=lambda x: f"{x:,}",
            help="Number of lines in the needle-in-haystack example"
        )
        
        # Store in session state for access in load_example
        st.session_state.example_size = example_size
        
        st.markdown("""
        **How RLM works:**
        1. Root LLM receives query + context
        2. Writes Python code in REPL
        3. REPL calls sub-LLMs for chunks
        4. Combines results for final answer
        """)
        
        # Clear button
        if st.button("🗑️ Clear All", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.sidebar.markdown('[Lawrence\'s Blog](https://lawrence.eti.br/)')
         
    # Main content area - adjusted column ratios
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("📝 Input")
        
        # Initialize session state for inputs if not present
        if "context_input" not in st.session_state:
            st.session_state.context_input = ""
        if "query_input" not in st.session_state:
            st.session_state.query_input = ""
        
        # Load example function (must be defined before button)
        def load_example():
            import random
            answer = str(random.randint(1000000, 9999999))
            
            # Get example size from session state (default 1000)
            num_lines = st.session_state.get("example_size", 1000)
            
            # Generate context with configurable size
            lines = []
            random_words = ["blah", "random", "text", "data", "content", "information", "sample"]
            for _ in range(num_lines):
                num_words = random.randint(3, 8)
                line_words = [random.choice(random_words) for _ in range(num_words)]
                lines.append(" ".join(line_words))
            
            # Insert magic number in the middle 40-60% range
            magic_position = random.randint(int(num_lines * 0.4), int(num_lines * 0.6))
            lines[magic_position] = f"The magic number is {answer}"
            
            st.session_state.context_input = "\n".join(lines)
            st.session_state.query_input = "I'm looking for a magic number. What is it?"
            st.session_state.example_answer = answer
            st.session_state.example_num_lines = num_lines
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            st.button("📋 Load Example", use_container_width=True, on_click=load_example)
        
        # Context input with key for proper state management
        context_input = st.text_area(
            "Context",
            height=200,
            placeholder="Enter context here...",
            help="The context that the RLM will analyze.",
            key="context_input"
        )
        
        # Query input with key
        query_input = st.text_input(
            "Query",
            placeholder="What question do you want to ask?",
            key="query_input"
        )
        
        # Show example info
        if "example_answer" in st.session_state:
            num_lines = st.session_state.get("example_num_lines", 1000)
            st.info(f"🔍 Hidden answer: **{st.session_state.example_answer}** ({num_lines:,} lines)")
        
        with col_btn2:
            run_disabled = not context_input or not query_input
            if st.button("🚀 Run RLM", disabled=run_disabled, type="primary", use_container_width=True):
                st.session_state.running = True
                st.session_state.context = context_input
                st.session_state.query = query_input
                st.session_state.model = model
                st.session_state.recursive_model = recursive_model
                st.session_state.max_iterations = max_iterations
                # Clear previous results
                if "result" in st.session_state:
                    del st.session_state["result"]
                if "logs" in st.session_state:
                    del st.session_state["logs"]
                st.rerun()
    
    with col2:
        st.subheader("📊 Execution Log")
        
        # Status placeholder at top
        status_placeholder = st.empty()
        
        # Scrollable container for logs
        log_container = st.container(height=500)
        
        if st.session_state.get("running", False):
            status_placeholder.warning("⏳ Running RLM...")
            
            try:
                # Create logger with both containers
                logger = StreamlitLogger(log_container, status_placeholder)
                
                # Create RLM with Streamlit logging
                rlm = StreamlitRLM_REPL(
                    streamlit_logger=logger,
                    model=st.session_state.model,
                    recursive_model=st.session_state.recursive_model,
                    max_iterations=st.session_state.max_iterations
                )
                
                # Run completion
                result = rlm.completion(
                    context=st.session_state.context,
                    query=st.session_state.query
                )
                
                st.session_state.result = result
                st.session_state.logs = logger.steps
                st.session_state.running = False
                st.rerun()
                
            except Exception as e:
                status_placeholder.error(f"❌ Error: {str(e)}")
                st.session_state.running = False
        
        elif "logs" in st.session_state:
            status_placeholder.success("✅ Completed!")
            
            # Re-render saved logs
            with log_container:
                for step in st.session_state.logs:
                    step_type = step["type"]
                    
                    if step_type == "system":
                        with st.expander(step["title"], expanded=False):
                            st.markdown(step["content"])
                    
                    elif step_type == "response":
                        with st.expander(step["title"], expanded=False):
                            st.markdown(step["content"])
                            if step.get("has_code"):
                                st.caption("⚙️ Contains REPL code")
                    
                    elif step_type == "code":
                        with st.expander(step["title"], expanded=False):
                            st.code(step["code"], language="python")
                            if step["output"]:
                                st.markdown(step["output"])
                    
                    elif step_type == "final":
                        st.success(f"### {step['title']}")
                        st.markdown(step["content"])
        
        else:
            with log_container:
                st.info("Enter context and query, then click **Run RLM** to see verbose execution.")

    # Footer
    st.markdown(
        """
        <div style="text-align: center; color: gray; font-size: 0.8em;">
            Made by <a href="https://www.linkedin.com/in/lawrenceteixeira/">Lawrence Teixeira</a> with ❤️ using <a href="https://github.com/alexzhang13/rlm-minimal">RLM Minimal example</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
