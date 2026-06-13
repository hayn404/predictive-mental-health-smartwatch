// Ambient module stub for onnxruntime-react-native.
// Installed via `expo prebuild` + `npx expo install onnxruntime-react-native`.
// The real types ship with the package; this stub silences the editor until
// the dependency is added to node_modules.
declare module 'onnxruntime-react-native' {
  export class Tensor {
    constructor(type: 'float32' | 'int64' | 'int32', data: Float32Array | BigInt64Array | Int32Array, dims: number[]);
    data: Float32Array | BigInt64Array | Int32Array;
    dims: number[];
  }
  export interface InferenceSession {
    run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
  }
  export const InferenceSession: {
    create(uri: string): Promise<InferenceSession>;
  };
}
