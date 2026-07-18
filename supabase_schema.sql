create extension if not exists pgcrypto;

create table if not exists public.analysis_runs (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    class_name text not null,
    class_id text not null,
    cms_year integer,
    selected_drugs jsonb not null default '[]'::jsonb,
    results_json jsonb not null,
    source_status jsonb not null default '{}'::jsonb
);

alter table public.analysis_runs enable row level security;
alter table public.analysis_runs force row level security;

revoke all on table public.analysis_runs from anon, authenticated;
grant all on table public.analysis_runs to service_role;

comment on table public.analysis_runs is
    'Server-written PharmaSignal AI proof-of-concept analysis records. No public RLS policy is intentionally defined.';

create table if not exists public.drug_class_catalog (
    class_id text not null,
    class_name text not null,
    class_type text not null,
    rela_source text not null default 'Resolve on selection',
    rela text not null default '',
    member_count integer check (member_count is null or member_count >= 0),
    example_members jsonb not null default '[]'::jsonb,
    search_text text not null default '',
    source_updated_at timestamptz not null,
    synced_at timestamptz not null default now(),
    primary key (class_id, rela_source, rela),
    constraint drug_class_catalog_examples_array
        check (jsonb_typeof(example_members) = 'array')
);

create index if not exists drug_class_catalog_class_type_idx
    on public.drug_class_catalog (class_type);
create index if not exists drug_class_catalog_rela_source_idx
    on public.drug_class_catalog (rela_source);
create index if not exists drug_class_catalog_class_name_lower_idx
    on public.drug_class_catalog (lower(class_name));
create index if not exists drug_class_catalog_member_count_idx
    on public.drug_class_catalog (member_count)
    where member_count is not null;

alter table public.drug_class_catalog enable row level security;
alter table public.drug_class_catalog force row level security;

revoke all on table public.drug_class_catalog from anon, authenticated;
grant all on table public.drug_class_catalog to service_role;

comment on table public.drug_class_catalog is
    'Server-managed search cache of the official RxClass catalog; not an independent drug database. No public RLS policy is intentionally defined.';
