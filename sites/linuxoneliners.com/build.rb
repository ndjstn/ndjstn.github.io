#!/usr/bin/env ruby
# frozen_string_literal: true

require "erb"
require "fileutils"
require "json"
require "cgi"
require "time"

ROOT = File.expand_path(__dir__)
CONTENT = JSON.parse(File.read(File.join(ROOT, "content", "lessons.json")))
DIST = File.join(ROOT, "dist")

FileUtils.rm_rf(DIST)
FileUtils.mkdir_p(DIST)
FileUtils.cp_r(File.join(ROOT, "assets"), DIST)

def render_template(name, locals)
  template = File.read(File.join(ROOT, "templates", "#{name}.erb"))
  context = TemplateContext.new(locals)
  ERB.new(template, trim_mode: "-").result(context.binding)
end

class TemplateContext
  def initialize(locals)
    @locals = locals
  end

  def binding
    super
  end

  def method_missing(name, *args)
    key = name.to_s
    if @locals.key?(key)
      value = @locals[key]
      return value.respond_to?(:call) ? value.call(*args) : value
    end

    super
  end

  def respond_to_missing?(name, include_private = false)
    @locals.key?(name.to_s) || super
  end

  def html_attr(value)
    CGI.escapeHTML(value.to_s)
  end
end

series_by_id = CONTENT["series"].to_h { |series| [series["id"], series] }

helpers = {
  "series_title" => lambda { |id| series_by_id.fetch(id).fetch("title") }
}

def render_page(body:, title:, description:, root_path:, asset_path:)
  render_template(
    "layout",
    {
      "body" => body,
      "page_title" => title,
      "meta_description" => description,
      "root_path" => root_path,
      "asset_path" => asset_path
    }
  )
end

index_body = render_template(
  "index",
  {
    "lessons" => CONTENT["lessons"],
    "series" => CONTENT["series"],
    "root_path" => "/",
    "asset_path" => "/",
    **helpers
  }
)

File.write(
  File.join(DIST, "index.html"),
  render_page(
    body: index_body,
    title: "#{CONTENT["site"]["name"]} | #{CONTENT["site"]["tagline"]}",
    description: CONTENT["site"]["description"],
    root_path: "/",
    asset_path: "/"
  )
)

CONTENT["lessons"].each do |lesson|
  lesson_dir = File.join(DIST, "lessons", lesson["slug"])
  FileUtils.mkdir_p(lesson_dir)

  body = render_template(
    "lesson",
    {
      "lesson" => lesson,
      "root_path" => "/",
      "asset_path" => "/",
      **helpers
    }
  )

  File.write(
    File.join(lesson_dir, "index.html"),
    render_page(
      body: body,
      title: "#{lesson["title"]} | #{CONTENT["site"]["name"]}",
      description: lesson["problem"],
      root_path: "/",
      asset_path: "/"
    )
  )
end

feed = {
  "generated_at" => Time.now.utc.iso8601,
  "site" => CONTENT["site"],
  "lessons" => CONTENT["lessons"].map do |lesson|
    lesson.slice("slug", "title", "series", "hook", "command", "danger", "shorts", "linkedin", "youtube")
  end
}

File.write(File.join(DIST, "feed.json"), JSON.pretty_generate(feed))
puts "Built #{CONTENT["lessons"].length} lessons into #{DIST}"
