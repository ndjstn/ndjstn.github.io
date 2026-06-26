#!/usr/bin/env ruby
# frozen_string_literal: true

require "erb"
require "fileutils"
require "json"
require "cgi"
require "time"

ROOT = File.expand_path(__dir__)
CONTENT = JSON.parse(File.read(File.join(ROOT, "content", "lessons.json")))
PACK_DIR = File.join(ROOT, "content", "packs")
LESSONS = CONTENT.fetch("lessons") + Dir.glob(File.join(PACK_DIR, "*.json")).sort.flat_map do |path|
  JSON.parse(File.read(path))
end
DIST = File.join(ROOT, "dist")
GENERATED_AT = Time.now.utc.iso8601
DEMO_ARTIFACTS = File.join(ROOT, "artifacts", "demos")

FileUtils.rm_rf(DIST)
FileUtils.mkdir_p(DIST)
FileUtils.cp_r(File.join(ROOT, "assets"), DIST)
FileUtils.mkdir_p(File.join(DIST, "exports"))
if Dir.exist?(DEMO_ARTIFACTS)
  FileUtils.cp_r(DEMO_ARTIFACTS, File.join(DIST, "demos"))
end

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

  def json_script(value)
    JSON.generate(value)
        .gsub("<", "\\u003c")
        .gsub(">", "\\u003e")
        .gsub("&", "\\u0026")
  end
end

series_by_id = CONTENT["series"].to_h { |series| [series["id"], series] }
demo_summary_path = File.join(DEMO_ARTIFACTS, "summary.json")
demo_summary = File.exist?(demo_summary_path) ? JSON.parse(File.read(demo_summary_path)) : nil
demo_by_slug = if demo_summary
                 demo_summary.fetch("results", []).to_h { |result| [result["slug"], result] }
               else
                 {}
               end

helpers = {
  "series_title" => lambda { |id| series_by_id.fetch(id).fetch("title") },
  "demo_for" => lambda { |slug| demo_by_slug[slug] },
  "demo_terminal" => lambda { |slug|
    path = File.join(DEMO_ARTIFACTS, slug, "terminal.txt")
    File.exist?(path) ? File.read(path) : nil
  }
}

def render_page(body:, title:, description:, root_path:, asset_path:)
  render_template(
    "layout",
    {
      "body" => body,
      "page_title" => title,
      "meta_description" => description,
      "root_path" => root_path,
      "asset_path" => asset_path,
      "site" => CONTENT["site"],
      "generated_at" => GENERATED_AT
    }
  )
end

index_body = render_template(
  "index",
  {
    "lessons" => LESSONS,
    "series" => CONTENT["series"],
    "site" => CONTENT["site"],
    "demo_summary" => demo_summary,
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

LESSONS.each do |lesson|
  lesson_dir = File.join(DIST, "lessons", lesson["slug"])
  FileUtils.mkdir_p(lesson_dir)

  body = render_template(
    "lesson",
    {
      "lesson" => lesson,
      "demo" => demo_by_slug[lesson["slug"]],
      "terminal_output" => helpers["demo_terminal"].call(lesson["slug"]),
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

CONTENT["series"].each do |series|
  series_dir = File.join(DIST, "series", series["id"])
  FileUtils.mkdir_p(series_dir)
  series_lessons = LESSONS.select { |lesson| lesson["series"] == series["id"] }

  body = render_template(
    "series",
    {
      "series_item" => series,
      "lessons" => series_lessons,
      "root_path" => "/",
      "asset_path" => "/",
      **helpers
    }
  )

  File.write(
    File.join(series_dir, "index.html"),
    render_page(
      body: body,
      title: "#{series["title"]} | #{CONTENT["site"]["name"]}",
      description: series["summary"],
      root_path: "/",
      asset_path: "/"
    )
  )
end

feed = {
  "generated_at" => GENERATED_AT,
  "site" => CONTENT["site"],
  "demo_summary" => demo_summary,
  "lessons" => LESSONS.map do |lesson|
    lesson.slice("slug", "title", "series", "hook", "command", "danger", "shorts", "linkedin", "youtube")
  end
}

File.write(File.join(DIST, "feed.json"), JSON.pretty_generate(feed))
shorts_export = LESSONS.map do |lesson|
  {
    "slug" => lesson["slug"],
    "title" => lesson["shorts"]["title"],
    "caption" => lesson["shorts"]["caption"],
    "duration_target_seconds" => lesson["shorts"]["duration_target_seconds"],
    "voiceover" => lesson["shorts"]["voiceover"],
    "youtube_playlist" => lesson["youtube"]["playlist"],
    "site_url" => "https://#{CONTENT["site"]["domain"]}/lessons/#{lesson["slug"]}/"
  }
end

File.write(File.join(DIST, "exports", "shorts.json"), JSON.pretty_generate(shorts_export))

analytics_events = {
  "generated_at" => GENERATED_AT,
  "privacy_position" => "No personal profiles or accounts are required. Events should be aggregate unless a future product requires explicit user consent.",
  "events" => [
    {
      "name" => "command_copy",
      "why_it_matters" => "Shows which commands have hands-on intent.",
      "properties" => ["label", "path"],
      "buyer_signal" => "Useful command pages can be ranked by demonstrated intent, not only page views."
    },
    {
      "name" => "export_click",
      "why_it_matters" => "Shows whether platform-production artifacts are being used.",
      "properties" => ["label", "path"],
      "buyer_signal" => "A repeatable content pipeline is more transferable when its usage is measurable."
    },
    {
      "name" => "lesson_related_click",
      "why_it_matters" => "Shows whether visitors move through a series.",
      "properties" => ["from_slug", "to_slug"],
      "buyer_signal" => "Higher internal movement supports course, newsletter, and product packaging."
    },
    {
      "name" => "outbound_video_click",
      "why_it_matters" => "Shows which lessons drive YouTube traffic.",
      "properties" => ["slug", "platform"],
      "buyer_signal" => "Cross-platform traffic loops increase resilience and saleability."
    }
  ]
}

File.write(File.join(DIST, "exports", "analytics-events.json"), JSON.pretty_generate(analytics_events))

linkedin_posts = LESSONS.map do |lesson|
  <<~POST
    ## #{lesson["title"]}

    #{lesson["hook"]}

    Command:
    `#{lesson["command"]}`

    #{lesson["problem"]}

    Question: #{lesson["linkedin"]["question"]}

    #{lesson["linkedin"]["cta"]}: https://#{CONTENT["site"]["domain"]}/lessons/#{lesson["slug"]}/
  POST
end

File.write(
  File.join(DIST, "exports", "linkedin-posts.md"),
  "# LinkedIn Post Seeds\n\nGenerated: #{GENERATED_AT}\n\n#{linkedin_posts.join("\n---\n\n")}"
)
puts "Built #{LESSONS.length} lessons into #{DIST}"
