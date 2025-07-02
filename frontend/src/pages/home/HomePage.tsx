import { useState } from "react";
import { Button } from "../../components/ui/button";

interface Prediction{
    class: string;
    confidence: number;
}

interface LayerData{
    shape: number[];
    values: number[][];
}

interface VisualizationData{
    [layerName: string]: LayerData;
}

interface WaveformData{
    values: number[];
    sample_rate: number;
    duration: number;
}

interface APIResponse{
    predictions: Prediction[];
    visualization: VisualizationData;
    input_spectrogram: LayerData;
    waveform: WaveformData;
}

const HomePage = () => {
    const [vizData, setVizData] = useState<APIResponse|null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [fileName, setFileName] = useState("");
    const [error, setError] = useState<string|null>(null);

    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event?.target.files?.[0];
        if(!file) return;
        setFileName(file.name);
        setIsLoading(true);
        setError(null);
        setVizData(null);

        const reader = new FileReader();
        reader.readAsArrayBuffer(file);
        reader.onload = async () => {
            const arrayBuffer = reader.result as ArrayBuffer;
            const base64String = btoa(new Uint8Array(arrayBuffer).reduce((data, byte) => {return data+String.fromCharCode(byte)}, ""));
            
        }
    }

    return <main className="min-h-screen bg-stone-50 p-8">
        <div className="mx-auto max-w-[60%]">
            <div className="mb-12 text-center">
                <h1 className="mb-4 text-4xl font-light tracking-light text-grey-900">
                    Audio CNN
                </h1>
                <p className="text-md mb-8 text-stone-600">
                    Upload a WAV file to check if its dog sound or cat sound
                </p>
                <div className="flex flex-col items-center">
                    <div className="relative inline-block">
                        <input type="file" accept=".wav" id="file-upload" disabled={isLoading} className="absolute inset-0 w-full cursor-pointer opacity-0"/>
                        <Button variant="outline" size="lg" className="border-stone-100" disabled={isLoading}>{isLoading?"Analyzing...": "Choose a file"}</Button>
                    </div>
                </div>
            </div>
        </div>
    </main>
}

export default HomePage;