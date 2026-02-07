from .agent_file import build_file_agent
from .agent_router import build_router_agent
from .agent_validator import build_validator_agent


def main():
    print("Choose: 1=file agent  2=router agent  3=validator agent")
    choice = input("> ").strip()

    # ---- OPTION 1: FILE AGENT ----
    if choice == "1":
        mode = input("mode (summarize/todos/rewrite): ").strip()
        path = input("file path: ").strip()

        app = build_file_agent()
        out = app.invoke({"mode": mode, "file_path": path})
        print(out)

    # ---- OPTION 2: ROUTER AGENT ----
    elif choice == "2":
        user = input(
            "Ask something (e.g. 'summarize:path' or 'OpenAI'): "
        ).strip()

        router = build_router_agent()
        out = router.invoke({"user_text": user})

        if out.get("error"):
            print("ERROR:", out["error"])
            return

        # OPTIONAL: auto-validate classification results
        if out.get("action") == "classify_text" and out.get("result"):
            validator = build_validator_agent()
            validated = validator.invoke({"entity": out["result"]})

            if validated.get("parsed"):
                print(validated["parsed"])
            else:
                print("ERROR:", validated.get("error"))
        else:
            print(out.get("result", out))

    # ---- OPTION 3: VALIDATOR AGENT (STANDALONE) ----
    elif choice == "3":
        while True:
            entity = input("Entity to classify: ").strip()
            if entity:
                break
            print("Please type something (e.g., OpenAI, Elon Musk, New York City).")

        app = build_validator_agent()
        out = app.invoke({"entity": entity})

        if out.get("parsed"):
            print(out["parsed"])
        else:
            print("ERROR:", out.get("error", out))



if __name__ == "__main__":
    main()
