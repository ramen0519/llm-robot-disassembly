"""Existing pure execution lineage checks; re-exported by the original runner."""

def _unambiguous_decisions(decisions):
    """Conflicting records for one request cannot establish execution lineage."""
    result, ambiguous = {}, set()
    for decision in decisions:
        identifier = decision.get('request', {}).get('request_id')
        if identifier is None:
            continue
        if identifier in result and result[identifier] != decision:
            ambiguous.add(identifier)
        else:
            result[identifier] = decision
    return {key: value for key, value in result.items() if key not in ambiguous}


def execution_bindings(decisions, report):
    """A sent decision is not necessarily the decision the worker accepted."""
    all_proposed = _unambiguous_decisions(decisions)
    proposed_ids = {d['request']['request_id'] for d in decisions
                    if d['response']['metadata'].get('generated_program_executed') is True}
    received = _unambiguous_decisions(report.get('decisions', []))
    accepted, discarded = [], []
    for attempt in report.get('attempts', []):
        identifier = attempt['request_id']
        if identifier not in proposed_ids:
            continue
        host = all_proposed.get(identifier, {}).get('response', {})
        worker = received.get(identifier, {}).get('response', {})
        host_metadata = host.get('metadata', {})
        skill_id, program_hash = (host_metadata.get('generated_skill_id'),
                                  host_metadata.get('generated_program_hash'))
        matched = (type(skill_id) is str and bool(skill_id)
            and type(program_hash) is str and bool(program_hash)
            and host_metadata.get('generated_program_executed') is True
            and all_proposed[identifier]['request'] == received.get(identifier, {}).get('request')
            and attempt.get('action_id') == host.get('action_id') == worker.get('action_id')
            and attempt.get('choice') == host.get('choice') == worker.get('choice')
            and attempt.get('raw', {}).get('chosen') == host.get('choice')
            and worker.get('metadata', {}).get('generated_program_executed') is True
            and worker.get('metadata', {}).get('generated_program_hash') == program_hash
            and worker.get('metadata', {}).get('generated_skill_id') == skill_id)
        (accepted if matched else discarded).append(attempt)
    return accepted, discarded


def _actual_selector_bound(host, worker, selections):
    """Bind an actual selector response to this exact accepted action request."""
    metadata = host['response']['metadata']
    worker_metadata = worker['response']['metadata']
    identity, program_hash = metadata['generated_skill_id'], metadata['generated_program_hash']
    matches = []
    for item in selections:
        packet, response = item.get('packet', {}), item.get('response', {})
        provider = response.get('metadata', {})
        cards = [card for card in packet.get('skills', [])
                 if card.get('skill_id') == identity and card.get('program_hash') == program_hash]
        if (packet.get('current_request') == host['request'] == worker['request']
                and packet.get('operation') == response.get('operation') == 'select'
                and packet.get('request_id') == response.get('request_id')
                and type(packet.get('request_id')) is str
                and packet.get('run_id') == host['request'].get('run_id')
                and packet.get('state_id') == response.get('state_id') == host['request'].get('state_id')
                and len(cards) == 1
                and response.get('origin') == 'actual_llm'
                and response.get('status') == 'selected'
                and response.get('selected_skill_id') == identity
                and (provider.get('llm_live') is True or provider.get('original_llm_live') is True)
                and metadata.get('generated_skill_selection_origin') ==
                    worker_metadata.get('generated_skill_selection_origin') == 'actual_llm'
                and metadata.get('generated_selector_response') == response ==
                    worker_metadata.get('generated_selector_response')
                and packet.get('context', {}).get('library_sha256') ==
                    metadata.get('generated_library_sha256') ==
                    worker_metadata.get('generated_library_sha256')
                and type(metadata.get('generated_library_sha256')) is str):
            matches.append(item)
    # Ambiguous multiple selector records are not silently counted as one fact.
    return len(matches) == 1


def physical_usage_audit(decisions, selections, report):
    """Attribute physical attempts and accepted PASS to each exact program.

    A selected new skill elsewhere in the episode cannot borrow another
    program's physics execution. Actual LLM attribution additionally requires
    the matching selector packet, provider origin and host/worker response.
    """
    accepted, _ = execution_bindings(decisions, report)
    hosts = _unambiguous_decisions(decisions)
    workers = _unambiguous_decisions(report.get('decisions', []))
    usage = {}
    for attempt in accepted:
        identifier = attempt['request_id']
        host, worker = hosts[identifier], workers[identifier]
        metadata = host['response']['metadata']
        identity, program_hash = metadata['generated_skill_id'], metadata['generated_program_hash']
        row = usage.setdefault(identity, {
            'program_hash': program_hash, 'physical_attempts': 0, 'physical_passes': 0,
            'request_ids': [], 'action_ids': [],
            'actual_llm_selected_physical_attempts': 0,
            'actual_llm_selected_physical_passes': 0,
            'actual_llm_selected_request_ids': []})
        if row['program_hash'] != program_hash:
            raise ValueError('generated_skill_identity_has_conflicting_program_hashes')
        passed = (attempt.get('accepted_for_transition') is True
                  and attempt.get('status') == 'PASS'
                  and attempt.get('deadline_exceeded') is False)
        row['physical_attempts'] += 1
        row['physical_passes'] += int(passed)
        row['request_ids'].append(identifier)
        row['action_ids'].append(attempt['action_id'])
        if _actual_selector_bound(host, worker, selections):
            row['actual_llm_selected_physical_attempts'] += 1
            row['actual_llm_selected_physical_passes'] += int(passed)
            row['actual_llm_selected_request_ids'].append(identifier)
    usage = {key: usage[key] for key in sorted(usage)}
    return {'physical_usage_by_skill': usage,
        'physically_executed_skill_ids': list(usage),
        'physical_pass_skill_ids': [key for key, row in usage.items() if row['physical_passes']],
        'actual_llm_selected_physically_executed_skill_ids': [key for key, row in usage.items()
            if row['actual_llm_selected_physical_attempts']],
        'actual_llm_selected_physical_pass_skill_ids': [key for key, row in usage.items()
            if row['actual_llm_selected_physical_passes']],
        'actual_llm_selected_physical_attempts': sum(row['actual_llm_selected_physical_attempts']
            for row in usage.values()),
        'actual_llm_selected_physical_passes': sum(row['actual_llm_selected_physical_passes']
            for row in usage.values())}
