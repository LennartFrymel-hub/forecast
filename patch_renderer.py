import pathlib

path = pathlib.Path("/Users/bartz/workspace/spotforecast2-safe/.venv/lib/python3.13/site-packages/quartodoc/renderers/md_renderer.py")
content = path.read_text()

new_content = content.replace("    # unsupported parts ----", """
    @dispatch
    def render(self, el: ds.DocstringSectionFunctions) -> str:
        return ""

    @dispatch
    def render(self, el: ds.DocstringSectionClasses) -> str:
        return ""

    @dispatch
    def render(self, el: ds.DocstringSectionAttributes) -> str:
        return ""

    # unsupported parts ----""")

path.write_text(new_content)
print("done")
