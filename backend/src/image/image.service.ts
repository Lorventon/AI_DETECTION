import { Injectable } from '@nestjs/common';
import axios from 'axios';
import * as FormData from 'form-data';

@Injectable()
export class ImageService {
  private readonly pythonApiUrl = process.env.PYTHON_API || 'http://model:5000';

  public async processImage(file: { buffer: Buffer; originalname: string }, params: any): Promise<any> {
    const formData = new FormData();
    formData.append('file', file.buffer, { filename: file.originalname });
    formData.append('front_door_area', params.front_door_area);
    formData.append('torch_width', params.torch_width);
    formData.append('torch_extrusion', params.torch_extrusion);
    formData.append('paint_cost_per_liter', params.paint_cost_per_liter);

    try {
      const response = await axios.post(`${this.pythonApiUrl}/process`, formData, {
        headers: formData.getHeaders(),
      });
      return response.data;
    } catch (error) {
      console.error('Error calling Python service:', error);
      throw new Error('Error processing image');
    }
  }
}
