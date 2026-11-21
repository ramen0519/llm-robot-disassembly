"""Bounded two-episode learning cycle with explicit, injectable stage boundaries.

No stage is allowed to promote a mock result into real physics. Backends own
typed contracts, persisted proofs and budgets. This coordinator owns ordering.
"""
from copy import deepcopy


def run_training_cycle(tasks, backend, *, held_out_ids=()):
    ids = [task['assembly_id'] for task in tasks]
    if not ids or len(ids) != len(set(ids)) or set(ids).intersection(held_out_ids):
        raise ValueError('invalid_or_leaking_training_split')
    records = []
    for index, task in enumerate(tasks):
        row = dict(assembly_id=task['assembly_id'], stages=[], registered_skill=None,
                   reuse_status='NOT_RUN', actual_effect='UNVERIFIED')
        records.append(row)
        backend.event('task_started', task=task['assembly_id'])
        # All choices in an episode use an immutable snapshot.
        frozen = backend.freeze('task_%02d_round1' % index)
        episode = backend.episode(task, 0, frozen)
        row['stages'].append(dict(stage='round1', result=episode))
        backend.collect(episode)
        if backend.halted:
            row['stop_reason'] = backend.halted
            break
        packet = backend.packet(task, episode)
        if packet is None:
            row['stages'].append(dict(stage='learning', status='NO_OBSERVATIONS'))
        else:
            lesson_result = backend.lesson(task, packet)
            row['stages'].append(dict(stage='initial_lesson', result=lesson_result))
            if lesson_result.get('ok'):
                lesson = lesson_result['lesson']
                previous, feedback = None, None
                for revision in range(2):
                    if backend.halted:
                        break
                    candidate_result = backend.candidate(task, packet, lesson, previous, feedback, revision)
                    row['stages'].append(dict(stage='candidate', revision=revision, result=candidate_result))
                    if not candidate_result.get('ok'):
                        previous = candidate_result.get('candidate') or {'rejected_result': candidate_result}
                        feedback = candidate_result
                        continue
                    candidate = candidate_result['candidate']
                    validation = backend.validate_candidate(task, candidate, lesson)
                    row['stages'].append(dict(stage='validation', revision=revision, result=validation))
                    if not validation.get('ok'):
                        previous, feedback = candidate, validation
                        continue
                    post = backend.lesson(task, packet, previous_lesson=lesson,
                                          validation=validation, candidate=candidate)
                    row['stages'].append(dict(stage='postvalidation_lesson', result=post))
                    if post.get('ok'):
                        # Registration reopens and verifies all native proof files.
                        card = backend.register(candidate, post['lesson'], validation)
                        row['registered_skill'] = card['skill_id']
                    break
        if backend.halted:
            row['stop_reason'] = backend.halted
            break
        frozen = backend.freeze('task_%02d_round2' % index)
        second = backend.episode(task, 1, frozen)
        row['stages'].append(dict(stage='round2', result=second))
        backend.collect(second)
        row['reuse_status'] = second.get('reuse_status', 'UNVERIFIED')
        backend.event('task_finished', summary=deepcopy(row))
        if backend.halted:
            row['stop_reason'] = backend.halted
            break
    final = backend.freeze('final')
    return dict(protocol='sequence_training_cycle_v1', tasks=records, frozen=final,
                stopped=bool(backend.halted), stop_reason=backend.halted,
                task_count=len(records), generalization='NOT_TESTED',
                performance_effect='UNVERIFIED')
