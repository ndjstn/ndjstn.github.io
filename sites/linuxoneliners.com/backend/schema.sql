-- Future Postgres schema for the sellable linuxoneliners.com backend.
-- Keep operator-platform data in a separate database.

create table if not exists users (
  id bigserial primary key,
  email text unique,
  display_name text,
  role text not null default 'user',
  status text not null default 'active',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists sessions (
  id bigserial primary key,
  user_id bigint references users(id) on delete cascade,
  visitor_id text not null,
  session_id text not null,
  first_seen_at timestamptz not null default now(),
  last_seen_at timestamptz not null default now(),
  source text,
  campaign text,
  variant text,
  landing_path text,
  user_agent text,
  ip_hash text
);

create table if not exists commands (
  id bigserial primary key,
  slug text unique not null,
  title text not null,
  command text not null,
  danger text not null,
  status text not null default 'draft',
  source text not null default 'owned',
  created_by bigint references users(id),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists comments (
  id bigserial primary key,
  command_id bigint references commands(id) on delete cascade,
  user_id bigint references users(id) on delete set null,
  parent_id bigint references comments(id) on delete cascade,
  body text not null,
  status text not null default 'pending',
  moderation_reason text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists moderation_queue (
  id bigserial primary key,
  object_type text not null,
  object_id bigint not null,
  status text not null default 'pending',
  reason text not null,
  severity text not null default 'medium',
  model_notes text,
  reviewer_notes text,
  created_at timestamptz not null default now(),
  reviewed_at timestamptz
);

create table if not exists campaigns (
  id bigserial primary key,
  source text not null,
  campaign text not null,
  variant text not null default 'a',
  landing_path text not null,
  platform_url text,
  status text not null default 'active',
  created_at timestamptz not null default now(),
  unique (source, campaign, variant, landing_path)
);

create table if not exists events (
  id bigserial primary key,
  received_at timestamptz not null default now(),
  event_name text not null,
  visitor_id text,
  session_id text,
  user_id bigint references users(id) on delete set null,
  path text,
  referrer text,
  source text,
  campaign text,
  variant text,
  properties jsonb not null default '{}',
  performance jsonb not null default '{}',
  viewport jsonb not null default '{}',
  connection jsonb not null default '{}'
);

create table if not exists conversions (
  id bigserial primary key,
  event_id bigint references events(id) on delete set null,
  visitor_id text,
  session_id text,
  user_id bigint references users(id) on delete set null,
  conversion_type text not null,
  amount_cents integer,
  currency text default 'USD',
  source text,
  campaign text,
  variant text,
  created_at timestamptz not null default now()
);

create table if not exists ad_placements (
  id bigserial primary key,
  name text unique not null,
  placement_type text not null,
  status text not null default 'draft',
  disclosure_text text,
  created_at timestamptz not null default now()
);

create table if not exists ad_events (
  id bigserial primary key,
  placement_id bigint references ad_placements(id) on delete cascade,
  event_name text not null,
  visitor_id text,
  session_id text,
  path text,
  source text,
  campaign text,
  variant text,
  revenue_cents integer,
  created_at timestamptz not null default now()
);

create index if not exists idx_events_campaign on events(source, campaign, variant);
create index if not exists idx_events_session on events(session_id);
create index if not exists idx_events_path on events(path);
create index if not exists idx_conversions_campaign on conversions(source, campaign, variant);
create index if not exists idx_comments_status on comments(status);
create index if not exists idx_moderation_status on moderation_queue(status);
