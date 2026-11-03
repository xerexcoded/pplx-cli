import pytest
from pplx_cli.evidence import Evidence, CitationError, grounded_answer


def test_grounded_answer_repairs_invalid_citations_and_reports_omissions():
    prompts = []
    def writer(prompt):
        prompts.append(prompt)
        return 'Claim [bogus]' if len(prompts) == 1 else 'Claim [L1]'
    items = [Evidence('L1', 'file:1', 'hello'), Evidence('L2', 'file:2', 'x' * 9000)]
    answer = grounded_answer('Question?', items, writer, context_tokens=3000, output_tokens=256)
    assert answer.content == 'Claim [L1]'
    assert answer.omitted == 1
    assert len(prompts) == 2
    assert all(len(prompt.encode()) <= 3000 - 256 for prompt in prompts)


def test_grounded_answer_failure_exposes_evidence():
    item = Evidence('L1', 'file:1', 'original excerpt')
    with pytest.raises(CitationError, match='original excerpt') as error:
        grounded_answer('Question?', [item], lambda prompt: 'claim [L2]')
    assert error.value.evidence == [item]


def test_insufficient_evidence_needs_no_citation():
    answer = grounded_answer('Question?', [Evidence('L1', 'loc', 'text')],
                             lambda prompt: 'Insufficient evidence.')
    assert answer.content == 'Insufficient evidence.'


def test_text_sync_update_and_deletion(tmp_path):
    from tests.test_wiki_workspace import FakeEmbeddingModel
    from pplx_cli.wiki import WikiWorkspace
    workspace = WikiWorkspace(tmp_path, embedding_model=FakeEmbeddingModel())
    source = tmp_path / 'notes.txt'
    source.write_text('Vector evidence', encoding='utf-8')
    assert workspace.sync()['indexed'] == 1
    assert workspace.search('vector')[0].source_type == 'text'
    source.write_text('Changed citation', encoding='utf-8')
    assert workspace.sync()['updated'] == 1
    source.unlink()
    assert workspace.sync()['removed'] == 1


def test_damaged_page_prevents_all_generation_and_writes(tmp_path, monkeypatch):
    from tests.test_wiki_workspace import FakeEmbeddingModel
    from pplx_cli.wiki import WikiWorkspace
    workspace = WikiWorkspace(tmp_path, embedding_model=FakeEmbeddingModel())
    (tmp_path / 'source.txt').write_text('Vector evidence', encoding='utf-8')
    workspace.sync()
    overview = workspace.wiki_dir / 'overview.md'
    overview.write_text('manual notes without markers', encoding='utf-8')
    def no_call(*args, **kwargs):
        pytest.fail('must validate all pages before model calls')
    monkeypatch.setattr('pplx_cli.wiki.workspace.query_chat_completion', no_call)
    with pytest.raises(ValueError, match='markers'):
        workspace.compile()
    assert overview.read_text(encoding='utf-8') == 'manual notes without markers'


def test_local_compile_processes_all_chunks_and_rejects_invalid_output(tmp_path, monkeypatch):
    import sys
    import types
    from tests.test_wiki_workspace import FakeEmbeddingModel
    from pplx_cli.api import ChatCompletion
    from pplx_cli.local import local_only_mode
    from pplx_cli.wiki import WikiWorkspace
    workspace = WikiWorkspace(tmp_path, embedding_model=FakeEmbeddingModel())
    (tmp_path / 'source.md').write_text('# Source\n\n' + '\n\n'.join(
        f'## Part {i}\n\nVector evidence {i}' for i in range(10)), encoding='utf-8')
    workspace.sync()
    monkeypatch.setitem(sys.modules, 'pplx_cli.ollama', types.SimpleNamespace(get_context_budget=lambda model: 4096))
    prompts = []
    def writer(prompt, **kwargs):
        prompts.append(prompt)
        return ChatCompletion(content='Summary [^source-1]', model='fake')
    monkeypatch.setattr('pplx_cli.wiki.workspace.query_chat_completion', writer)
    with local_only_mode():
        workspace.compile(provider='ollama')
    assert any('Vector evidence 9' in prompt for prompt in prompts)
    overview = workspace.wiki_dir / 'overview.md'
    before = overview.read_text(encoding='utf-8')
    monkeypatch.setattr('pplx_cli.wiki.workspace.query_chat_completion',
                        lambda *args, **kwargs: ChatCompletion(content='Bad [^source-99]', model='fake'))
    with local_only_mode(), pytest.raises(CitationError):
        workspace.compile(provider='ollama')
    assert overview.read_text(encoding='utf-8') == before
