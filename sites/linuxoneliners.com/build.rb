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
GENERATED_AT = Time.now.utc.iso8601

FileUtils.rm_rf(DIST)
FileUtils.mkdir_p(DIST)
FileUtils.cp_r(File.join(ROOT, "assets"), DIST)
FileUtils.mkdir_p(File.join(DIST, "exports"))

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
      "asset_path" => asset_path,
      "site" => CONTENT["site"],
      "generated_at" => GENERATED_AT
    }
  )
end

index_body = render_template(
  "index",
  {
    "lessons" => CONTENT["lessons"],
    "series" => CONTENT["series"],
    "site" => CONTENT["site"],
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
  "generated_at" => GENERATED_AT,
  "site" => CONTENT["site"],
  "lessons" => CONTENT["lessons"].map do |lesson|
    lesson.slice("slug", "title", "series", "hook", "command", "danger", "shorts", "linkedin", "youtube")
  end
}

File.write(File.join(DIST, "feed.json"), JSON.pretty_generate(feed))
shorts_export = CONTENT["lessons"].map do |lesson|
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

linkedin_posts = CONTENT["lessons"].map do |lesson|
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
puts "Built #{CONTENT["lessons"].length} lessons into #{DIST}"
