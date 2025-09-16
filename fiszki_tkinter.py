import os
import sys
import webview
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Global variable for pywebview window
pywebview_window = None


def save_content(file_path, html_content, encoding):
    """Saves the raw HTML content to preserve formatting."""
    try:
        with open(file_path, "w", encoding=encoding) as f:
            f.write(html_content)
        print(f"✅ Content saved successfully to {file_path} with encoding {encoding}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")


def extract_and_save(window_like, file_path, encoding_used):
    """Gets HTML from editor and saves it (parity with PySide2)."""
    global pywebview_window
    win = window_like or pywebview_window
    if not win:
        print("❌ No webview window available")
        return

    html_content = win.evaluate_js("getEditorContent()")
    if html_content is None:
        print("❌ Could not read editor content")
        return

    save_content(file_path, html_content, encoding_used)
    win.evaluate_js("alert('Zmiany zostały zapisane!')")


class EditorAPI:
    """Bridge for JS ↔ Python calls."""

    def __init__(self, file_path, encoding_used, window_accessor):
        self.file_path = file_path
        self.encoding_used = encoding_used
        self.window_accessor = window_accessor

    def save_direct(self, html_content):
        save_content(self.file_path, html_content, self.encoding_used)
        return "Zapisano zmiany!"

    def trigger_save(self):
        win = self.window_accessor()
        extract_and_save(win, self.file_path, self.encoding_used)
        return "OK"


def create_html_editor(file_path, file_content, choice, encoding_used):
    """Creates the HTML editor in a pywebview window."""
    global pywebview_window

    file_content_js = file_content.replace("\\", "\\\\").replace("`", "\\`").replace("\n", "<br>")

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
           

            html, body {{
                width: 100%;
                height: 100%;
                margin: 0;
                padding: 0;
                font-family: Arial, Times New Roman, Courier New, Georgia, Verdana, sans-serif; 
                background-color: #2B2D30;
                display: flex;
                justify-content: center;
                align-items: center;
            }}

            .editor-container {{
                width: 95%;
                height: 90%;
                gap: 12px;
                max-width: 900px;
                background: #525252;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                display: flex;
                flex-direction: column;
                max-height: 90vh;
            }}

            .toolbar {{
                display: flex;
                align-items: center;
                background: #BDBDBD;
                padding: 10px;
                border-bottom: 1px solid #ddd;
                border-radius: 8px;
            }}

            .toolbar-left {{
                display: flex;
                gap: 8px;
            }}

            .toolbar-left button {{
                margin-right: 8px;
            }}

            .toolbar-left button:last-child {{
                margin-right: 0;
            }}

            .toolbar button {{
                background-color: #1f6aa5;
                color: white;
                border: none;
                padding: 6px 12px;
                font-size: 14px;
                cursor: pointer;
                border-radius: 5px;
                transition: 0.3s;
            }}

            .toolbar button:hover {{
                background-color: #FF0000;
            }}
            

            
            .search-highlight {{
    background-color: yellow;
}}

            .search-container {{
                display: flex;
                align-items: center;
                margin-left: auto;
                gap: 8px;
            }}
            
            

            .search-container input {{
                padding: 5px;
                border-radius: 5px;
                border: 1px solid #ccc;
                width: 130px;
            }}

            #editor {{
                border: 1px solid #3498db;
                border-radius: 8px;                
                flex-grow: 1;
                padding: 15px;
                white-space: pre-wrap;
                background: white; Optional: remove gray background */
                border-radius: 5px;
                font-family: Arial, Times New Roman, Courier New, Georgia, Verdana, sans-serif; 
                font-size: 16px;
                line-height: 1.5;
                outline: none;
                overflow: auto;
            }}
        </style>
        <script>
            let lastIndex = -1;

            function formatText(command, color) {{
                if (command === "forecolor") {{
                    document.execCommand(command, false, color);
                }} else {{
                    document.execCommand(command, false, null);
                }}
            }}

            function getEditorContent() {{
                const editor = document.getElementById("editor");
                // Clone the editor to avoid modifying what's displayed
                const clone = editor.cloneNode(true);
                // Remove search highlights
                const highlights = clone.querySelectorAll("span.search-highlight");
                highlights.forEach(span => {{
                    const parent = span.parentNode;
                    while (span.firstChild) {{
                        parent.insertBefore(span.firstChild, span);
                    }}
                    parent.removeChild(span);
                }});
                return clone.innerHTML;
            }}

            function copyContent() {{
         clearHighlights();
    const selection = window.getSelection();
    if (selection.rangeCount > 0 && !selection.isCollapsed) {{
        try {{
            const successful = document.execCommand('copy');
            if (successful) {{
                alert('Zaznaczony tekst skopiowany!');
            }} else {{
                const textToCopy = selection.toString();
                prompt("Skopiuj tekst poniżej:", textToCopy);
            }}
            // Clear highlights after copying
            
        }} catch {{
            const textToCopy = selection.toString();
            prompt("Skopiuj tekst poniżej:", textToCopy);
           
        }}
    }} else {{
        alert('Zaznacz tekst do skopiowania.');
    }}
}}

            function triggerSave() {{
                const editorContent = getEditorContent();
                if (window.pywebview) {{
                    window.pywebview.api.save_direct(editorContent).then((msg) => {{
                        alert(msg);
                    }});
                }} else {{
                    alert("Webview API not available");
                }}
            }}

            // Bind Ctrl+C / Cmd+C to Kopiuj button
            document.addEventListener('keydown', function(e) {{
                const editor = document.getElementById("editor");
                const selection = window.getSelection();
                if ((e.ctrlKey || e.metaKey) && e.key === 'c') {{
                    if (selection.rangeCount > 0 && !selection.isCollapsed && editor.contains(selection.anchorNode)) {{
                        document.getElementById("kopiujBtn").click();
                        e.preventDefault();
                    }}
                }}
                // Bind Ctrl+S / Cmd+S to Zapisz button
                if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {{
                    e.preventDefault();
                    document.getElementById("zapiszBtn").click();
                }}
            }});

            function applySelectedColor() {{
                const colorPicker = document.getElementById("favcolor");
                const colorAButton = document.getElementById("colorAButton");
                const selectedColor = colorPicker.value;
                document.execCommand('forecolor', false, selectedColor);
            }}
function searchContentBackward() {{
    const editor = document.getElementById("editor");
    const searchTerm = document.getElementById("searchTerm").value;
    const matchCounter = document.getElementById("matchCounter");
    if (!searchTerm) {{
        matchCounter.textContent = "0/0";
        return;
    }}

    const selection = window.getSelection();
    selection.removeAllRanges();

    const regex = new RegExp(searchTerm, "gi");
    const walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT, null, false);
    let matches = [];
    while (walker.nextNode()) {{
        const node = walker.currentNode;
        let match;
        while ((match = regex.exec(node.nodeValue)) !== null) {{
            matches.push({{ node, start: match.index, end: match.index + match[0].length }});
        }}
    }}

    if (matches.length === 0) {{
        matchCounter.textContent = "0/0";
        return;
    }}

    lastIndex = (lastIndex - 1 + matches.length) % matches.length;
    const match = matches[lastIndex];

    const range = document.createRange();
    range.setStart(match.node, match.start);
    range.setEnd(match.node, match.end);
    selection.addRange(range);

    const rect = range.getBoundingClientRect();
    window.scrollTo({{ top: rect.top + window.scrollY - window.innerHeight / 2, behavior: "smooth" }});

    matchCounter.textContent = `${{lastIndex + 1}}/${{matches.length}}`;
}}
           function searchContent() {{
    const editor = document.getElementById("editor");
    const searchTerm = document.getElementById("searchTerm").value;
    const matchCounter = document.getElementById("matchCounter");
    if (!searchTerm) {{
        matchCounter.textContent = "0/0";
        return;
    }}

    const selection = window.getSelection();
    selection.removeAllRanges();

    const regex = new RegExp(searchTerm, "gi");
    const walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT, null, false);
    let matches = [];
    while (walker.nextNode()) {{
        const node = walker.currentNode;
        let match;
        while ((match = regex.exec(node.nodeValue)) !== null) {{
            matches.push({{ node, start: match.index, end: match.index + match[0].length }});
        }}
    }}

    if (matches.length === 0) {{
        matchCounter.textContent = "0/0";
        return;
    }}

    lastIndex = (lastIndex + 1) % matches.length;
    const match = matches[lastIndex];

    const range = document.createRange();
    range.setStart(match.node, match.start);
    range.setEnd(match.node, match.end);
    selection.addRange(range);

    const rect = range.getBoundingClientRect();
    window.scrollTo({{ top: rect.top + window.scrollY - window.innerHeight / 2, behavior: "smooth" }});

    // Update match counter
    matchCounter.textContent = `${{lastIndex + 1}}/${{matches.length}}`;
}}

function hsearchContent() {{
    const editor = document.getElementById("editor");
    const searchTerm = document.getElementById("searchTerm").value;
    if (!searchTerm) return;

    clearHighlights(); // remove previous highlights

    const regex = new RegExp(searchTerm, "gi");

    // Use a tree walker to iterate over text nodes
    const walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT, null, false);
    let nodes = [];
    while (walker.nextNode()) {{
        nodes.push(walker.currentNode);
    }}

    nodes.forEach(node => {{
        let nodeValue = node.nodeValue;
        let parent = node.parentNode;
        let match;
        let span;

        let frag = document.createDocumentFragment();
        let lastIndex = 0;

        while ((match = regex.exec(nodeValue)) !== null) {{
            // Text before match
            if (match.index > lastIndex) {{
                frag.appendChild(document.createTextNode(nodeValue.slice(lastIndex, match.index)));
            }}
            // Matched text
            span = document.createElement("span");
            span.className = "search-highlight";
            span.textContent = match[0];
            frag.appendChild(span);

            lastIndex = match.index + match[0].length;
        }}

        // Remaining text after last match
        if (lastIndex < nodeValue.length) {{
            frag.appendChild(document.createTextNode(nodeValue.slice(lastIndex)));
        }}

        parent.replaceChild(frag, node);
    }});
}}
            
            function clearHighlights() {{
                const editor = document.getElementById("editor");
                const highlights = editor.querySelectorAll("span.search-highlight");
                highlights.forEach(span => {{
                    const parent = span.parentNode;
                    while (span.firstChild) {{
                        parent.insertBefore(span.firstChild, span);
                    }}
                    parent.removeChild(span);
                }});
            }}
                        
            function changeFont(font) {{
                document.execCommand("fontName", false, font);
            }}

            function adjustEditorHeight() {{
                const editor = document.getElementById("editor");
                editor.style.height = (window.innerHeight * 0.7) + "px";
            }}

            window.onload = function() {{
                document.getElementById("editor").innerHTML = `{file_content_js}`;
                adjustEditorHeight();
                window.addEventListener("resize", adjustEditorHeight);


                        // Set initial color to the color picker's current value
            colorAButton.style.color = colorPicker.value;

            // Update color when the picker changes
            colorPicker.addEventListener("input", function() {{
                colorAButton.style.color = colorPicker.value;
            }});
            }};
        </script>
    </head>
    <body>
        <div class="editor-container">
            <div class="toolbar">
                <div class="toolbar-left">
                    <button onclick="formatText('bold')"><b>B</b></button>
                    <button onclick="formatText('italic')"><em>I</em></button>
                    <button onclick="formatText('underline')"><u>U</u></button>
                    <button id="colorAButton" onclick="applySelectedColor()" style="font-weight: bold;">A</button>


                                        <input type="color" id="favcolor" value="#ff0000">
                    <script>
                        const colorPicker = document.getElementById("favcolor");
                        colorPicker.addEventListener("input", function() {{
                            const selectedColor = colorPicker.value;
                        colorAButton.style.color = selectedColor;  // update button text color
                    }});
                    </script>
                    <select id="fontSelect" onchange="changeFont(this.value)">
                      <option value="Arial">Arial</option>
                      <option value="Times New Roman">Times New Roman</option>
                      <option value="Courier New">Courier New</option>
                      <option value="Georgia">Georgia</option>
                      <option value="Verdana">Verdana</option>
                    </select>

                    <button id="kopiujBtn" onclick="copyContent()">Kopuj</button>
                    <button id="zapiszBtn" onclick="triggerSave()">Zapisz</button>


                </div>
<div class="search-container" style="position: relative; display: inline-block;">
    <input type="text" id="searchTerm" placeholder="Szukaj..." style="padding-right: 90px;"> <!-- extra space for buttons -->
    
   <!-- Match counter -->
<span id="matchCounter" style="
    position: absolute;
    top: 50%;
    right: 80px;  /* adjust to sit nicely next to your line */
    transform: translateY(-50%);
    font-size: 12px;
    color: #5F5F62;
">0/0</span>
     <!-- Vertical line inside input -->
    <div style="
        position: absolute;
        top: 5px;
        bottom: 5px;
        right: 70px;  /* position before buttons */
        width: 1px;
        background-color: #ccc;
    "></div>
    <!-- Up button -->
    <button onclick="hsearchContent(); searchContentBackward();" 
            style="
                position: absolute;
                right: 52px;  /* before the X button */
                top: 50%;
                transform: translateY(-50%);
                border: none;
                background: transparent;
                cursor: pointer;
                font-size: 16px;
                color: #5F5F62;
                padding: 0;
            ">∧</button>

    <!-- Down button -->
    <button onclick="hsearchContent(); searchContent();" 
            style="
                position: absolute;
                right: 30px;  /* just before the X button */
                top: 50%;
                transform: translateY(-50%);
                border: none;
                background: transparent;
                cursor: pointer;
                font-size: 16px;
                color: #5F5F62;
                padding: 0;
           
            ">∨</button>

    <!-- Clear button -->
    <button id="clearHighlightsBtn" onclick="clearHighlights()" 
            style="
                position: absolute;
                right: 10px;
                top: 50%;
                transform: translateY(-50%);
                border: none;
                background: transparent;
                cursor: pointer;
                font-size: 12px;
                color: #5F5F62;
                padding: 0;
            ">X</button>
</div>
<script>
    // Get the input element
    const searchInput = document.getElementById('searchTerm');

    // Listen for the Enter key
    searchInput.addEventListener('keydown', function(event) {{
        if (event.key === 'Enter') {{     // check if Enter was pressed
            event.preventDefault();        // prevent default form submission if inside a form
            hsearchContent();              // your first function
            searchContent();               // your second function
        }}
    }});
</script>
                
                
            </div>
            <div id="editor" contenteditable="true"></div>
        </div>
    </body>
    </html>
    """

    # Create the webview window
    title = f"Fiszka: {choice}"
    window_accessor = lambda: pywebview_window
    api = EditorAPI(file_path=file_path, encoding_used=encoding_used, window_accessor=window_accessor)
    pywebview_window = webview.create_window(title, html=html_template, js_api=api, width=900, height=600)
    webview.start(debug=False)


def load_file_content(choice):
    """Loads file content and opens the HTML editor."""
    file_path = os.path.join("fiszki", choice + ".txt")

    if os.path.exists(file_path):
        encodings = ["utf-8", "Windows-1250", "ISO-8859-2"]
        content = None
        encoding_used = None

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    content = file.read()
                    encoding_used = encoding
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            print(f"❌ Failed to decode {file_path}")
            return

        create_html_editor(file_path, content, choice, encoding_used)
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("")
        create_html_editor(file_path, "", choice, "utf-8")


if __name__ == "__main__":
    choice = "test" if len(sys.argv) < 2 else sys.argv[1]
    load_file_content(choice)
