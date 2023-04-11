import papermill as pm


def run_notebook():
    pm.execute_notebook(
        'voc_presentation_notebook.ipynb',
        'voc_presentation_notebook_output.ipynb'
        )


if __name__ == "__main__":
    run_notebook()
