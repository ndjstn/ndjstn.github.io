#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"

path = ARGV.fetch(0, "ops/valuation/portfolio.example.json")
portfolio = JSON.parse(File.read(path))

portfolio.fetch("sites").each do |site|
  metrics = site.fetch("monthly_metrics")
  valuation = site.fetch("valuation")
  profit = metrics.fetch("net_profit").to_f

  low = profit * valuation.fetch("low_multiple").to_f
  mid = profit * valuation.fetch("mid_multiple").to_f
  high = profit * valuation.fetch("high_multiple").to_f

  puts site.fetch("domain")
  puts "  monthly net profit: $#{format('%.2f', profit)}"
  puts "  rough valuation:    $#{format('%.2f', low)} - $#{format('%.2f', high)}"
  puts "  midpoint:           $#{format('%.2f', mid)}"
  puts "  note:               #{valuation.fetch('notes')}"
end
