/**
 * Narrow compatibility shim for DeepSeek Harness developer-preview tool calls.
 *
 * Some long turns wrap an otherwise valid tool object as
 * `{ "arguments": "{...}" }`.  The official agent loop parses only the outer
 * JSON layer, so Bash then receives no `command`.  This shim unwraps at most
 * three such compatibility layers before the registered tool validates the
 * call.  It never invents a command and never repairs truncated/invalid JSON.
 */

import { appendFileSync } from 'node:fs'

export const name = 'truthinsightbench-tool-argument-normalizer'

function plainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function expectedKeys(name) {
  if (name === 'bash') return ['command']
  if (name === 'str_replace_editor') return ['command', 'path']
  return []
}

function normalize(name, original) {
  const expected = expectedKeys(name)
  if (expected.length === 0) return { value: original, layers: 0 }
  let value = original
  let layers = 0
  while (layers < 3) {
    if (typeof value === 'string') {
      try {
        value = JSON.parse(value)
        layers += 1
        continue
      } catch {
        break
      }
    }
    if (!plainObject(value)) break
    if (expected.some(key => Object.hasOwn(value, key))) break
    const keys = Object.keys(value)
    if (!Object.hasOwn(value, 'arguments')) break
    if (keys.some(key => key !== 'arguments' && key !== 'description')) break
    value = value.arguments
    layers += 1
  }
  return { value, layers }
}

function audit(record) {
  const path = process.env.TIB_DSH_NORMALIZER_LOG
  if (!path) return
  appendFileSync(path, `${JSON.stringify(record)}\n`, { encoding: 'utf8' })
}

export function apply(ctx) {
  ctx.on('tools/execute', async (exec, next) => {
    const { value, layers } = normalize(exec.name, exec.arguments)
    if (layers > 0) {
      // The developer-preview runtime keeps the dispatch object mutable until
      // result notification.  Replacing this field is intentionally narrow;
      // the normalized object still passes the tool's own schema validator.
      exec.arguments = value
      audit({ call_id: String(exec.callId), tool: exec.name, unwrapped_layers: layers })
    }
    return next()
  })
}
