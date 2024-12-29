from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

def load_pdf(data):
    print("Loading pdf file...")
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
        )
    document = loader.load()
    print("Done.")
    return document