import os
import sys
import subprocess


def limpar_tela():
    os.system("cls" if os.name == "nt" else "clear")


def rodar_script(script_nome: str, tipo_alvo: str):
    comando = [sys.executable, script_nome, "--tipo", tipo_alvo]
    print(
        f"\n[Executando]: {' '.join(comando)}\n{'-'*60}"
    )

    try:
        resultado = subprocess.run(comando, check=True)
        print(f"{'-'*60}\n[Sucesso]: {script_nome} finalizou com sucesso.")
    except subprocess.CalledProcessError:
        print(
            f"{'-'*60}\n[Erro Crítico]: Ocorreu uma falha na execução de {script_nome}."
        )
    except KeyboardInterrupt:
        print(f"\n{'-'*60}\n[Aviso]: Execução interrompida.")

    input("\nPressione [ENTER] para voltar ao menu principal...")


def main():
    while True:
        limpar_tela()
        print("=" * 65)
        print("     SISTEMA CENTRAL DE GERENCIAMENTO DO PIPELINE DO TCC")
        print("=" * 65)
        print(" [1]  Executar Pipeline Bronze       --> [BINÁRIO]")
        print(" [2]  Executar Pipeline Bronze       --> [MULTICLASSE]")
        print(" [3]  Executar Pipeline Prata        --> [BINÁRIO]")
        print(" [4]  Executar Pipeline Prata        --> [MULTICLASSE]")
        print(" [5]  Executar Treinamento de Modelos--> [BINÁRIO]")
        print(" [6]  Executar Treinamento de Modelos--> [MULTICLASSE]")
        print(" [7]  Executar Testes e Avaliação    --> [BINÁRIO]")
        print(" [8]  Executar Testes e Avaliação    --> [MULTICLASSE]")
        print("-" * 65)
        print(" [0]  Sair do Sistema")
        print("=" * 65)

        opcao = input("Selecione uma opção de execução: ").strip()

        if opcao == "1":
            rodar_script("bronze.py", "binario")
        elif opcao == "2":
            rodar_script("bronze.py", "multiclasse")
        elif opcao == "3":
            rodar_script("prata.py", "binario")
        elif opcao == "4":
            rodar_script("prata.py", "multiclasse")
        elif opcao == "5":
            rodar_script("treino.py", "binario")
        elif opcao == "6":
            rodar_script("treino.py", "multiclasse")
        elif opcao == "7":
            rodar_script("teste.py", "binario")
        elif opcao == "8":
            rodar_script("teste.py", "multiclasse")
        elif opcao == "0":
            print("\nEncerrando...")
            break
        else:
            print("\nOpção inválida! Tente novamente.")
            time_sleep = input(
                "Pressione [ENTER] para continuar..."
            ) 


if __name__ == "__main__":
    main()
