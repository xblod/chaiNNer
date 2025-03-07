import { Type } from '@chainner/navi';
import { EdgeData, InputId, OutputId } from '../common-types';
import { FunctionDefinition } from '../types/function';
import { parseTargetHandle } from '../util';
import type { Edge } from 'reactflow';

export const getConnectedInputs = (
    nodeId: string,
    edges: readonly Edge<EdgeData>[]
): Set<InputId> => {
    const targetedInputs = edges
        .filter((e) => e.target === nodeId && e.targetHandle)
        .map((e) => parseTargetHandle(e.targetHandle!).inputId);
    return new Set(targetedInputs);
};

export const getFirstPossibleInput = (fn: FunctionDefinition, type: Type): InputId | undefined => {
    return fn.schema.inputs.find((i) => i.hasHandle && fn.canAssignInput(i.id, type))?.id;
};
export const getFirstPossibleOutput = (
    outputFn: FunctionDefinition,
    inputFn: FunctionDefinition,
    inputId: InputId
): OutputId | undefined => {
    for (const [id, type] of outputFn.outputDefaults) {
        if (inputFn.canAssignInput(inputId, type)) {
            return id;
        }
    }
    return undefined;
};
