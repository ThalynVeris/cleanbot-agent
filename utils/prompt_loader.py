from utils.config_handler import prompts_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompts_config["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompts] 配置项缺失: {str(e)}")
        raise e

    try:
        return open(system_prompt_path, "r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts] 加载系统提示词失败: {str(e)}")
        raise e

def load_rag_prompts():
    try:
        rag_config_path = get_abs_path(prompts_config["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompts] 配置项缺失: {str(e)}")
        raise e
    try:
        return open(rag_config_path, "r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts] 解析RAG总结提示词失败: {str(e)}")
        raise e

def load_report_prompts():
    try:
        report_config_path = get_abs_path(prompts_config["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts] 配置项缺失: {str(e)}")
        raise e
    try:
        return open(report_config_path, "r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts] 解析报告生成提示词失败: {str(e)}")
        raise e

if __name__ == "__main__":
    print(load_system_prompts())
    print(load_rag_prompts())
    print(load_report_prompts())