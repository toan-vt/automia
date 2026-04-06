from pathlib import Path
from .common import register_logger, get_logger, LLMClient, LLMClientProvider, GraphState, ExperimentConfig
from .agents import MutatorAgent, CodeGenAgent, ExecutorAgent, ResultReaderAgent, CodeFixAgent, ExplorerAgent, ExploiterAgent
from .tools import DatabaseTool, EmbeddingTool
from argparse import ArgumentParser
from multiprocessing import Process
import time

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="/tmp/automia")
    parser.add_argument("--experiment-dir", type=str, default="label-only-mia")
    parser.add_argument("--timeout", type=int, default=5*60)
    parser.add_argument("--model-name", type=str, default="qwen")
    parser.add_argument("--model-name-thinking", type=str, default="qwen-thinking")
    parser.add_argument("--base-url", type=str, default="http://localhost:9800/v1")
    parser.add_argument("--base-url-thinking", type=str, default="http://localhost:9800/v1")
    parser.add_argument("--embedding-model-name", type=str, default="embedding")
    parser.add_argument("--embedding-base-url", type=str, default="http://localhost:9700/v1")
    parser.add_argument("--db-table-name", type=str, default="dev")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--backend", type=str, default="file", choices=["postgres", "file"], help="Storage backend ('postgres' or 'file')")
    parser.add_argument("--provider", type=str, default="vllm", choices=["vllm", "openai", "google"], help="LLM provider ('vllm', 'openai', 'google')")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Running AutoMIA with args: {args}")
    output_dir = args.output_dir
    budget = args.budget
    model_name = args.model_name
    experiment_dir = args.experiment_dir
    timeout = args.timeout
    db_table_name = args.db_table_name
    thinking_mode = args.thinking
    backend = args.backend
    base_url = args.base_url
    model_name_thinking = args.model_name_thinking
    base_url_thinking = args.base_url_thinking
    embedding_model_name = args.embedding_model_name
    embedding_base_url = args.embedding_base_url
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    register_logger("system", Path(output_dir) / "log_system.log")
    data_dir = "database"
    if backend == "file":
        data_dir = Path(output_dir) / "database"
        data_dir.mkdir(parents=True, exist_ok=True)
    system_logger = get_logger("system")
    register_logger("llm", Path(output_dir) / "log_llm.log")

    # Load experiment config from experiment-dir/config.yaml
    config = ExperimentConfig.from_yaml(experiment_dir)

    if args.provider == "vllm":
        llm_provider = LLMClientProvider.VLLM
    elif args.provider == "openai":
        llm_provider = LLMClientProvider.OPENAI
    elif args.provider == "google":
        llm_provider = LLMClientProvider.GOOGLE
    else:
        raise ValueError(f"Invalid provider: {args.provider}")


    # Initialize LLM clients
    llm_client = LLMClient(model_name=model_name, provider=llm_provider, base_url=base_url)
    if thinking_mode:
        llm_thinking_client = LLMClient(model_name=model_name_thinking, provider=llm_provider, base_url=base_url_thinking, temperature=0.6)

    ## RUN THE SOTA  ==============================================================================================================
    embedding_tool = EmbeddingTool(model_name=embedding_model_name, vllm_api_base=embedding_base_url, provider=LLMClientProvider.VLLM)
    local_db = DatabaseTool(table_name=db_table_name, backend=backend, data_dir=data_dir, embedding_tool=embedding_tool)
    if local_db.get_num_experiments() == 0: # if the database is empty, run the SOTA experiment
        executor_agent = ExecutorAgent(f"{experiment_dir}/template.py", output_dir=output_dir, timeout=timeout, bash_env_script_path=f"{experiment_dir}/env.sh", db=local_db)
        if thinking_mode:
            result_reader_agent = ResultReaderAgent(llm_thinking_client, config)
        else:
            result_reader_agent = ResultReaderAgent(llm_client, config)

        state = GraphState(
            idea=config.high_level_idea,
            design_justification=config.design_justification,
            implementation_instruction="",
            code_block=config.example_python_code,
            error_flag=False,
            error_message="",
            fix_attempts=0,
            experiment_result={},
            analysis_summary="",
            experiment_id=-1
        )
        executor_agent(state)
        result_reader_agent(state)
        local_db.insert_experiment(state.idea, state.design_justification, state.code_block, state.experiment_result["auc_score"], state.experiment_result["tpr_1_score"], state.experiment_result["tpr_5_score"], state.experiment_result["combined_score"], state.analysis_summary, parent_id=state.experiment_id)

    ## RUN ITERATIVELY: 1 explorer then 3 exploiter =================================================================================
    if thinking_mode:
        explorer_agent = ExplorerAgent(llm_thinking_client, local_db, config)
        exploiter_agent = ExploiterAgent(llm_thinking_client, local_db, config)
    else:
        explorer_agent = ExplorerAgent(llm_client, local_db, config)
        exploiter_agent = ExploiterAgent(llm_client, local_db, config)
    coder_agent = CodeGenAgent(llm_client, config)
    code_fix_agent = CodeFixAgent(llm_client, config)
    executor_agent = ExecutorAgent(f"{experiment_dir}/template.py", output_dir=output_dir, timeout=timeout, bash_env_script_path=f"{experiment_dir}/env.sh", db=local_db)
    if thinking_mode:
        result_reader_agent = ResultReaderAgent(llm_thinking_client, config)
    else:
        result_reader_agent = ResultReaderAgent(llm_client, config)

    while local_db.get_num_experiments() < budget:
        try:
            iteration = local_db.get_num_experiments()
            state = GraphState(
                idea="",
                design_justification="",
                implementation_instruction="",
                code_block="",
                error_flag=False,
                error_message="",
                fix_attempts=0,
                experiment_result={},
                analysis_summary="",
                experiment_id=-1
            )
            if iteration % 4 == 0:
                explorer_agent(state)
            else:
                exploiter_agent(state)

            coder_agent(state)
            executor_agent(state)

            if not state.error_flag:
                result_reader_agent(state)
                local_db.insert_experiment(state.idea, state.design_justification, state.code_block, state.experiment_result["auc_score"], state.experiment_result["tpr_1_score"], state.experiment_result["tpr_5_score"], state.experiment_result["combined_score"], state.analysis_summary, parent_id=state.experiment_id)
            else:
                while state.fix_attempts < 3:
                    code_fix_agent(state)
                    executor_agent(state)
                    if not state.error_flag:
                        result_reader_agent(state)
                        local_db.insert_experiment(state.idea, state.design_justification, state.code_block, state.experiment_result["auc_score"], state.experiment_result["tpr_1_score"], state.experiment_result["tpr_5_score"], state.experiment_result["combined_score"], state.analysis_summary, parent_id=state.experiment_id)
                        break
                    else:
                        if "<TIMEOUT>" in state.error_message:
                            state.fix_attempts += 2 # only fix once if timeout

        except Exception as e:
            system_logger.error(f"Error in main loop: {e}")
            continue

if __name__ == "__main__":
    main()
