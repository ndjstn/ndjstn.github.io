from flask import Flask, render_template_string, request, redirect, url_for
import os
from datetime import datetime

app = Flask(__name__)

POSTS_DIR = './_posts'
ASSETS_DIR = './assets'

@app.route('/')
def index():
    return redirect(url_for('create_post'))

@app.route('/create_post', methods=['GET', 'POST'])
def create_post():
    if request.method == 'POST':
        title = request.form['title']
        tags = request.form['tags']
        content = request.form['content']
        post_date = request.form['date']
        images = request.files.getlist('images')
        
        # Generate the post filename and directory
        post_filename = f"{post_date}-{title.replace(' ', '-').lower()}.md"
        post_filepath = os.path.join(POSTS_DIR, post_filename)
        
        # Handle images
        image_urls = []
        if images and images[0].filename != '':
            image_dir = os.path.join(ASSETS_DIR, title.replace(' ', '-').lower())
            os.makedirs(image_dir, exist_ok=True)
            for image in images:
                image_path = os.path.join(image_dir, image.filename)
                image.save(image_path)
                image_url = f"/{image_path.replace(os.path.sep, '/')}"
                image_urls.append(image_url)

        # Create the post content based on the template
        post_content = f"""---
title: "{title}"
date: {post_date} 00:00:00 +0000
tags: [{tags}]
---

<p>{content}</p>
"""

        # Insert images into content
        for idx, image_url in enumerate(image_urls):
            image_tag = f'<p><a href="{image_url}"><img src="{image_url}" alt="Image {idx+1}"/></a></p>'
            post_content = post_content.replace(f'[image_{idx+1}]', image_tag)

        # Append the script for comments
        post_content += """
<script src="https://utteranc.es/client.js"
        repo="ndjstn/ndjstn.github.io"
        issue-term="url"
        theme="github-dark"
        crossorigin="anonymous"
        async>
</script>
"""

        # Save the post content to the file
        with open(post_filepath, 'w') as file:
            file.write(post_content)
        
        return redirect(url_for('index'))

    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Create Post</title>
            <link href="https://cdn.quilljs.com/1.3.6/quill.snow.css" rel="stylesheet">
            <script src="https://cdn.quilljs.com/1.3.6/quill.js"></script>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/jqueryui/1.12.1/jquery-ui.css">
            <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/jqueryui/1.12.1/jquery-ui.min.js"></script>
            <style>
                body {
                    background-color: #121212;
                    color: #ffffff;
                }
                input, textarea {
                    background-color: #333333;
                    color: #ffffff;
                }
                .ql-toolbar.ql-snow {
                    border: 1px solid #444444;
                }
                .ql-container.ql-snow {
                    border: 1px solid #444444;
                    color: #ffffff;
                }
                .ui-widget-content {
                    background: #333333;
                    border: 1px solid #444444;
                    color: #ffffff;
                }
                .ui-autocomplete {
                    max-height: 100px;
                    overflow-y: auto;
                    overflow-x: hidden;
                }
                .image-button {
                    background-color: #444444;
                    color: #ffffff;
                    border: none;
                    padding: 10px;
                    margin: 5px;
                    cursor: pointer;
                }
                .image-button:hover {
                    background-color: #555555;
                }
            </style>
        </head>
        <body>
            <h1>Create a New Post</h1>
            <form action="/create_post" method="post" enctype="multipart/form-data">
                <label for="title">Title:</label><br>
                <input type="text" id="title" name="title" required><br><br>
                
                <label for="tags">Tags (comma-separated):</label><br>
                <input type="text" id="tags" name="tags" required><br><br>

                <label for="date">Date:</label><br>
                <input type="text" id="datepicker" name="date" required><br><br>

                <label for="content">Content:</label><br>
                <div id="editor-container" style="height: 300px;"></div>
                <textarea id="content" name="content" style="display:none;"></textarea><br><br>
                
                <label for="images">Images:</label><br>
                <input type="file" id="images" name="images" multiple><br><br>
                
                <div id="image-buttons"></div>
                
                <input type="submit" value="Create Post">
            </form>
            <br>
            <a href="/">Back to Home</a>

            <script>
                $( function() {
                    $( "#datepicker" ).datepicker({ dateFormat: 'yy-mm-dd' });
                });

                var quill = new Quill('#editor-container', {
                    theme: 'snow'
                });

                // Auto-suggest for tags
                var availableTags = ["First post", "Test Post", "Technology", "Science", "Health", "Travel", "Food"];
                $( "#tags" ).autocomplete({
                    source: availableTags
                });

                var imageCount = 0;

                document.querySelector('#images').addEventListener('change', function(event) {
                    var imageButtons = document.querySelector('#image-buttons');
                    imageButtons.innerHTML = '';
                    imageCount = event.target.files.length;

                    for (var i = 0; i < imageCount; i++) {
                        var button = document.createElement('button');
                        button.textContent = 'Insert Image ' + (i + 1);
                        button.className = 'image-button';
                        button.setAttribute('data-image-index', i + 1);
                        button.addEventListener('click', function(event) {
                            event.preventDefault();
                            var index = event.target.getAttribute('data-image-index');
                            var range = quill.getSelection();
                            if (range) {
                                quill.insertText(range.index, '[image_' + index + ']', 'user');
                            }
                        });
                        imageButtons.appendChild(button);
                    }
                });

                document.querySelector('form').onsubmit = function() {
                    var content = document.querySelector('textarea[name=content]');
                    content.value = quill.root.innerHTML;
                };
            </script>
        </body>
        </html>
    ''')

if __name__ == '__main__':
    app.run(debug=True)
