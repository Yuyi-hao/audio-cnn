"use client";
import { useState } from "react";
import { Button } from "../../components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

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
            const api_url = "";
            try{
                const arrayBuffer = reader.result as ArrayBuffer;
                const base64String = btoa(new Uint8Array(arrayBuffer).reduce((data, byte) => {return data+String.fromCharCode(byte)}, ""));
                const response = await fetch(api_url, {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({audio_data: base64String})
                });
                if(!response.ok){
                    throw new Error(`API error: ${response.statusText}`);
                }
                const data:APIResponse = await response.json();
                setVizData(data);
            }
            catch(err){
                setError(err instanceof Error? err.message: "An Unknown error occurred");
            }
            finally{
                setIsLoading(false);
            }
        };
        reader.onerror = () => {
            setError("Failed to read file");
            setIsLoading(false);
        }
    };

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
                        <input type="file" accept=".wav" id="file-upload" disabled={isLoading} className="absolute inset-0 w-full cursor-pointer opacity-0" onChange={handleFileChange}/>
                        <Button variant="outline" size="lg" className="border-stone-100" disabled={isLoading}>{isLoading?"Analyzing...": "Choose a file"}</Button>
                    </div>
                    {true && (<Badge variant="secondary" className="mt-4 bg-stone-200 text-stone-800">{fileName}</Badge>)}
                </div>
            </div>
            {error && (<Card className="mb-8 border-red-200 bg-red-300">
                <CardContent>
                    <p className="text-red-600">Error: {error}</p>
                </CardContent>
            </Card>)}
        </div>
    </main>
}

export default HomePage;