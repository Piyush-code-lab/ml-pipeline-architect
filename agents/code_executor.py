from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


def run_code(code: str) -> tuple[bool, str]:
    try:
        exec(code)
        return True, ""

    except Exception as e:
        return False, str(e)


def execute_with_correction(code: str) -> dict:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Fix the provided Python code.

            Return only the corrected code.
            Do not include explanations.
            Do not use markdown code fences.
            """
        ),
        (
            "human",
            """Code:
            {code}

            Error:
            {error}
            """
        )
    ])

    parser = StrOutputParser()

    correction_chain = prompt | llm | parser

    max_attempts = 3
    current_code = code
    last_error = None

    for attempt in range(1, max_attempts + 1):

        success, error = run_code(current_code)

        if success:
            return {
                "final_code": current_code,
                "success": True,
                "attempts": attempt,
                "error": None
            }

        last_error = error

        current_code = correction_chain.invoke({
            "code": current_code,
            "error": error
        })

    return {
        "final_code": current_code,
        "success": False,
        "attempts": max_attempts,
        "error": last_error
    }

if __name__ == "__main__":
    broken_code = """
import pandas as pd
df = pd.DataFrame({'a': [1,2,3]})
print(df['b'])
"""
    result = execute_with_correction(broken_code)
    print(f"Success: {result['success']}")
    print(f"Attempts: {result['attempts']}")
    print(f"Error: {result['error']}")
    print(f"Final Code:\n{result['final_code']}")