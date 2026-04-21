import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, redirect, render_template_string, request, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

POSTS_DIR = Path('./_posts')
ASSETS_DIR = Path('./assets')
SITE_TIMEZONE = ZoneInfo('America/Chicago')


def slugify(value):
    slug = re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-')
    return slug or 'post'


def normalize_tags(raw_tags):
    tags = []
    seen = set()

    for candidate in raw_tags.split(','):
        tag = candidate.strip()
        tag_key = tag.lower()

        if tag and tag_key not in seen:
            tags.append(tag)
            seen.add(tag_key)

    return tags


def build_post_content(title, post_date, tags, content, image_urls):
    post_datetime = datetime.combine(post_date, datetime.min.time(), SITE_TIMEZONE)

    front_matter = [
        '---',
        f'title: {json.dumps(title)}',
        f"date: {post_datetime.strftime('%Y-%m-%d %H:%M:%S %z')}"
    ]

    if tags:
        front_matter.append('tags:')
        front_matter.extend([f'  - {json.dumps(tag)}' for tag in tags])

    front_matter.append('---')

    body = content.strip()

    for index, image_url in enumerate(image_urls, start=1):
        image_tag = (
            f'<p><a href="{image_url}"><img src="{image_url}" alt="Image {index}"></a></p>'
        )
        body = body.replace(f'[image_{index}]', image_tag)

    return '\n\n'.join(['\n'.join(front_matter), body]).strip() + '\n'


def save_images(images, post_slug):
    valid_images = [image for image in images if image and image.filename]

    if not valid_images:
        return []

    image_dir = ASSETS_DIR / post_slug
    image_dir.mkdir(parents=True, exist_ok=True)

    image_urls = []

    for image in valid_images:
        filename = secure_filename(image.filename)

        if not filename:
            continue

        image_path = image_dir / filename
        image.save(image_path)
        image_urls.append(f'/{image_path.as_posix()}')

    return image_urls

@app.route('/')
def index():
    return redirect(url_for('create_post'))

@app.route('/create_post', methods=['GET', 'POST'])
def create_post():
    if request.method == 'POST':
        title = request.form['title'].strip()
        tags = normalize_tags(request.form['tags'])
        content = request.form['content'].strip()
        post_date = datetime.strptime(request.form['date'], '%Y-%m-%d').date()
        images = request.files.getlist('images')

        post_slug = slugify(title)
        post_filename = f"{post_date.strftime('%Y-%m-%d')}-{post_slug}.md"
        post_filepath = POSTS_DIR / post_filename
        image_urls = save_images(images, post_slug)
        post_content = build_post_content(title, post_date, tags, content, image_urls)

        with post_filepath.open('w', encoding='utf-8') as file:
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
